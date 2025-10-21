"""
Orchestrates the workflow of the trading agents using LangGraph.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from langgraph.graph import StateGraph, END

from src.agents.data_structures import AgentDecision, MarketData
from src.agents.portfolio import PortfolioManagementAgent
from src.agents.risk import RiskManagementAgent
from src.agents.sentiment import SentimentAnalysisAgent
from src.agents.technical import TechnicalAnalysisAgent
from src.config.settings import settings
from src.data.pipeline import DataPipeline
from src.llm.batch_client import BatchRequestManager

logger = logging.getLogger(__name__)


class AgentState(dict):
    """
    Represents the state of the agentic graph. Inherits from dict.
    """
    market_data: MarketData
    decisions: Dict[str, AgentDecision]
    portfolio_state: Dict[str, Any]
    final_decision: AgentDecision
    error: str = ""
    signal_sources: Dict[str, str] = {}  # Track signal sources (LOCAL, LLM, etc.)
    performance_metrics: Dict[str, Any] = {}  # Track performance metrics


class Orchestrator:
    """
    Manages the agentic workflow, starting from data fetching.
    """

    def __init__(
        self,
        data_pipeline: DataPipeline,
        technical_agent: TechnicalAnalysisAgent,
        sentiment_agent: SentimentAnalysisAgent,
        risk_agent: RiskManagementAgent,
        portfolio_agent: PortfolioManagementAgent,
        state_manager: Any,  # For portfolio state
        trigger_system: Any = None,  # Optional trigger system for event-driven analysis
    ):
        self.data_pipeline = data_pipeline
        self.technical_agent = technical_agent
        self.sentiment_agent = sentiment_agent
        self.risk_agent = risk_agent
        self.portfolio_agent = portfolio_agent
        self.state_manager = state_manager
        self.trigger_system = trigger_system
        self.batch_manager = BatchRequestManager(technical_agent.llm_client)

        # Track signal generation metrics
        self.orchestrator_metrics = {
            "total_workflows": 0,
            "local_signal_workflows": 0,
            "llm_signal_workflows": 0,
            "hybrid_workflows": 0,
            "trigger_driven_workflows": 0,
            "workflow_errors": 0,
        }

        self.workflow = self._build_graph()

    def _build_graph(self):
        from langgraph.constants import START
        workflow = StateGraph(AgentState)
        workflow.add_node("technical_analysis", self.run_technical_analysis)
        workflow.add_node("sentiment_analysis", self.run_sentiment_analysis)
        workflow.add_node("batch_process", self.run_batch_process)
        workflow.add_node("risk_management", self.run_risk_management)
        workflow.add_node("portfolio_management", self.run_portfolio_management)
        # Run technical and sentiment in parallel to queue requests, then batch process
        workflow.add_edge(START, "technical_analysis")
        workflow.add_edge(START, "sentiment_analysis")
        workflow.add_edge("technical_analysis", "batch_process")
        workflow.add_edge("sentiment_analysis", "batch_process")
        workflow.add_edge("batch_process", "risk_management")
        workflow.add_edge("risk_management", "portfolio_management")
        workflow.add_edge("portfolio_management", END)
        return workflow.compile()

    async def run_technical_analysis(self, state: AgentState) -> dict:
        market_data = state.get("market_data")
        if not market_data:
            return {"error": "Market data not found"}

        # Delegate to the agent's analyze method which handles local/hybrid/LLM logic
        try:
            decision = await self.technical_agent.analyze(market_data)
            decisions = state.get("decisions", {})
            decisions["technical"] = decision

            # Track signal source from decision metadata
            signal_sources = state.get("signal_sources", {})
            signal_source = decision.supporting_data.get("signal_source", "UNKNOWN")
            signal_sources["technical"] = signal_source

            # Update orchestrator metrics based on source
            if signal_source == "LOCAL":
                self.orchestrator_metrics["local_signal_workflows"] += 1
            elif signal_source == "LLM":
                self.orchestrator_metrics["llm_signal_workflows"] += 1
            elif "escalation_info" in decision.supporting_data:
                # This was escalated from local to LLM
                self.orchestrator_metrics["hybrid_workflows"] += 1

            return {
                "decisions": decisions,
                "signal_sources": signal_sources,
                "batched_technical": False  # Processed directly via analyze()
            }
        except Exception as e:
            print(f"Error in technical analysis: {e}")
            return {"error": f"Technical analysis failed: {e}"}

    async def run_sentiment_analysis(self, state: AgentState) -> dict:
        market_data = state.get("market_data")
        if not market_data:
            return {"error": "Market data not found"}

        user_prompt = await self.sentiment_agent.get_user_prompt(market_data)
        await self.batch_manager.add_request(
            "sentiment",
            self.sentiment_agent.config.model_name,
            user_prompt,
            self.sentiment_agent.get_system_prompt()
        )
        return {"batched_sentiment": True}

    async def run_batch_process(self, state: AgentState) -> dict:
        market_data = state.get("market_data")
        if not market_data:
            return {"error": "Market data not found"}

        decisions = state.get("decisions", {})
        signal_sources = state.get("signal_sources", {})

        # Process all queued LLM batch requests in a single batch call
        # This handles both technical (if not local) and sentiment analysis
        results = await self.batch_manager.process_batch()

        # Process technical analysis result if it came from LLM batch
        if "technical" not in decisions and "technical" in results:
            decision = self.technical_agent.create_decision(market_data, results["technical"])
            decisions["technical"] = decision
            signal_sources["technical"] = "LLM"
        elif "technical" in decisions and "technical" not in signal_sources:
            # Technical was already processed locally
            signal_sources["technical"] = "LOCAL"

        # Process sentiment analysis result
        if "sentiment" not in decisions and "sentiment" in results:
            decision = self.sentiment_agent.create_decision(market_data, results["sentiment"])
            decisions["sentiment"] = decision
            signal_sources["sentiment"] = "LLM"

        # Check for hybrid mode escalations
        if "technical" in decisions:
            tech_decision = decisions["technical"]
            if "escalation_info" in tech_decision.supporting_data:
                # This was an escalated decision from local to LLM
                signal_sources["technical"] = "ESCALATED"
                self.orchestrator_metrics["hybrid_workflows"] += 1

        return {
            "decisions": decisions,
            "signal_sources": signal_sources
        }

    async def run_risk_management(self, state: AgentState) -> dict:
        market_data = state.get("market_data")
        decisions = state.get("decisions", {})
        portfolio_state = state.get("portfolio_state", {})
        if not market_data:
            return {"error": "Market data not found"}

        decision = await self.risk_agent.analyze(
            market_data, proposed_decisions=decisions, portfolio_state=portfolio_state
        )
        decisions["risk"] = decision
        return {"decisions": decisions}

    async def run_portfolio_management(self, state: AgentState) -> dict:
        market_data = state.get("market_data")
        decisions = state.get("decisions", {})
        portfolio_state = state.get("portfolio_state", {})
        if not market_data:
            return {"error": "Market data not found"}

        final_decision = await self.portfolio_agent.analyze(
            market_data, agent_decisions=decisions, portfolio_state=portfolio_state
        )

        # Add portfolio decision to decisions dict so it appears in API response
        decisions["portfolio"] = final_decision

        return {
            "decisions": decisions,
            "final_decision": final_decision
        }

    async def run(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> AgentState:
        # Calculate days requested by user
        days_requested = (end_date - start_date).days

        # Use all requested days for historical data
        # (with minimum for indicators that need baseline)
        historical_periods = max(days_requested, 30)  # Min 30 for indicator calculations

        market_data = await self.data_pipeline.fetch_and_process_data(
            symbol, start_date, end_date,
            historical_periods=historical_periods  # Pass full period instead of default 10
        )
        if not market_data:
            self.orchestrator_metrics["workflow_errors"] += 1
            # Provide more specific error message for symbol issues
            # (Note: We're already in an async context, so don't use asyncio.run)
            from src.data.symbol_validator import SymbolValidator
            symbol_validator = SymbolValidator()
            is_valid, validation_error = await symbol_validator.validate_symbol(symbol)

            if validation_error:
                return AgentState(error=validation_error.message)
            else:
                return AgentState(error=f"Failed to fetch data for {symbol} - no market data available")

        portfolio_state = self.state_manager.get_portfolio_state() or {
            "cash": settings.portfolio.STARTING_CASH,
            "positions": {},
        }
        initial_state = AgentState(
            market_data=market_data,
            decisions={},
            portfolio_state=portfolio_state,
            signal_sources={},
            performance_metrics={}
        )

        try:
            # Update total workflows counter
            self.orchestrator_metrics["total_workflows"] += 1

            final_state_dict = await self.workflow.ainvoke(initial_state)
            final_state = AgentState(final_state_dict)

            # Add performance metrics to the final state
            final_state["performance_metrics"] = {
                "orchestrator_metrics": self.orchestrator_metrics.copy(),
                "technical_agent_metrics": self.technical_agent.get_performance_metrics() if hasattr(self.technical_agent, 'get_performance_metrics') else {},
            }

            return final_state

        except Exception as e:
            self.orchestrator_metrics["workflow_errors"] += 1
            return AgentState(error=f"Workflow execution failed: {e}")

    def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """
        Get orchestrator performance metrics.

        Returns:
            Dict[str, Any]: Orchestrator metrics including signal source distribution
        """
        metrics = self.orchestrator_metrics.copy()

        # Add calculated metrics
        if metrics["total_workflows"] > 0:
            metrics["local_signal_percentage"] = (
                metrics["local_signal_workflows"] / metrics["total_workflows"] * 100
            )
            metrics["llm_signal_percentage"] = (
                metrics["llm_signal_workflows"] / metrics["total_workflows"] * 100
            )
            metrics["hybrid_percentage"] = (
                metrics["hybrid_workflows"] / metrics["total_workflows"] * 100
            )
            metrics["error_rate"] = (
                metrics["workflow_errors"] / metrics["total_workflows"] * 100
            )

        # Add technical agent metrics
        if hasattr(self.technical_agent, 'get_performance_metrics'):
            metrics["technical_agent_metrics"] = self.technical_agent.get_performance_metrics()

        return metrics

    def reset_orchestrator_metrics(self):
        """Reset orchestrator metrics."""
        self.orchestrator_metrics = {
            "total_workflows": 0,
            "local_signal_workflows": 0,
            "llm_signal_workflows": 0,
            "hybrid_workflows": 0,
            "workflow_errors": 0,
        }

        # Reset technical agent metrics
        if hasattr(self.technical_agent, 'reset_performance_metrics'):
            self.technical_agent.reset_performance_metrics()

    async def analyze_symbol(
        self,
        symbol: str,
        trigger_event: Any = None,
        force_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze a symbol, optionally triggered by an event.

        Args:
            symbol: Stock symbol to analyze
            trigger_event: Optional trigger event that initiated this analysis
            force_analysis: Force analysis even if recently analyzed

        Returns:
            Analysis results including decisions and metadata
        """
        try:
            # Update metrics if trigger-driven
            if trigger_event:
                self.orchestrator_metrics["trigger_driven_workflows"] += 1
                logger.info(f"Starting trigger-driven analysis for {symbol}: {trigger_event.trigger_type.value}")

            # Use recent date range for real-time analysis
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # 30 days for indicators

            # Check decision cache if not forced
            if not force_analysis and self.trigger_system:
                cached_decision = await self.trigger_system.decision_ttl.get_decision(symbol, "full_analysis")
                if cached_decision:
                    logger.info(f"Using cached analysis for {symbol}")
                    return {
                        'symbol': symbol,
                        'cached': True,
                        'decision': cached_decision.decision,
                        'confidence': cached_decision.confidence,
                        'timestamp': cached_decision.timestamp.isoformat()
                    }

            # Run the standard analysis workflow
            final_state = await self.run(symbol, start_date, end_date)

            if 'error' in final_state:
                return {
                    'symbol': symbol,
                    'error': final_state['error'],
                    'trigger_info': trigger_event.to_dict() if trigger_event else None
                }

            # Extract analysis results
            results = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'trigger_driven': trigger_event is not None,
                'decisions': {},
                'final_decision': None,
                'signal_sources': final_state.get('signal_sources', {}),
                'performance_metrics': final_state.get('performance_metrics', {})
            }

            # Extract individual decisions
            decisions = final_state.get('decisions', {})
            for agent_name, decision in decisions.items():
                results['decisions'][agent_name] = {
                    'action': decision.action,
                    'confidence': decision.confidence,
                    'reasoning': decision.reasoning,
                    'metadata': decision.supporting_data
                }

            # Extract final decision
            if 'final_decision' in final_state:
                final_decision = final_state['final_decision']
                results['final_decision'] = {
                    'action': final_decision.action,
                    'confidence': final_decision.confidence,
                    'reasoning': final_decision.reasoning
                }

            # Add trigger information
            if trigger_event:
                results['trigger_info'] = {
                    'type': trigger_event.trigger_type.value,
                    'severity': trigger_event.severity.value,
                    'confidence': trigger_event.confidence,
                    'description': trigger_event.description,
                    'data': trigger_event.data
                }

            # Cache the result if trigger system is available
            if self.trigger_system and 'final_decision' in results:
                final_action = results['final_decision']['action']
                final_confidence = results['final_decision']['confidence']

                # Determine decision severity based on action and confidence
                if final_action in ['STRONG_BUY', 'STRONG_SELL'] or final_confidence >= 0.8:
                    from src.events.trigger_detector import TriggerSeverity
                    severity = TriggerSeverity.HIGH
                else:
                    severity = TriggerSeverity.MEDIUM

                await self.trigger_system.decision_ttl.cache_decision(
                    symbol=symbol,
                    decision_type="full_analysis",
                    decision=final_action,
                    severity=severity,
                    confidence=final_confidence,
                    metadata={
                        'trigger_type': trigger_event.trigger_type.value if trigger_event else None,
                        'decisions': results['decisions'],
                        'signal_sources': results['signal_sources']
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Error in trigger-driven analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'trigger_info': trigger_event.to_dict() if trigger_event else None
            }
