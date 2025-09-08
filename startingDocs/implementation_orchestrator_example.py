import asyncio
from typing import Dict, List
from dataclasses import asdict

class AgentOrchestrator:
    def __init__(self, message_bus, state_manager):
        self.message_bus = message_bus
        self.state_manager = state_manager
        self.agents: Dict[str, BaseAgent] = {}

    def register_agent(self, agent):
        """Register an agent with the orchestrator"""
        self.agents[agent.config.name] = agent
        # Subscribe agent to message bus
        self.message_bus.subscribe(agent.config.name, agent.handle_message)

    async def sequential_analysis(self, symbol: str, market_data) -> List:
        """Execute agents in sequence: Technical -> Sentiment -> Risk -> Portfolio"""
        decisions = []

        # Technical analysis first
        if 'technical' in self.agents:
            technical_decision = await self.agents['technical'].analyze(market_data)
            decisions.append(technical_decision)
            await self.state_manager.set_state(
                f'technical_{symbol}',
                asdict(technical_decision)
            )

        # Sentiment analysis
        if 'sentiment' in self.agents:
            sentiment_decision = await self.agents['sentiment'].analyze(market_data)
            decisions.append(sentiment_decision)
            await self.state_manager.set_state(
                f'sentiment_{symbol}',
                asdict(sentiment_decision)
            )

        # Risk analysis
        if 'risk' in self.agents:
            risk_decision = await self.agents['risk'].analyze(market_data)
            decisions.append(risk_decision)

        # Final portfolio decision
        if 'portfolio' in self.agents:
            portfolio_decision = await self.agents['portfolio'].analyze(market_data)
            decisions.append(portfolio_decision)

        return decisions

    async def parallel_analysis(self, symbol: str, market_data) -> List:
        """Execute technical and sentiment agents in parallel"""
        # Create tasks for parallel execution
        tasks = []

        if 'technical' in self.agents:
            tasks.append(self.agents['technical'].analyze(market_data))
        if 'sentiment' in self.agents:
            tasks.append(self.agents['sentiment'].analyze(market_data))

        # Execute in parallel
        parallel_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Store results for other agents
        for i, result in enumerate(parallel_results):
            if not isinstance(result, Exception):
                agent_name = list(self.agents.keys())[i]
                await self.state_manager.set_state(
                    f'{agent_name}_{symbol}',
                    asdict(result)
                )

        # Risk analysis (depends on other results)
        risk_decision = None
        if 'risk' in self.agents:
            risk_decision = await self.agents['risk'].analyze(market_data)

        # Final portfolio decision
        portfolio_decision = None
        if 'portfolio' in self.agents:
            portfolio_decision = await self.agents['portfolio'].analyze(market_data)

        # Combine all results
        all_results = [r for r in parallel_results if not isinstance(r, Exception)]
        if risk_decision:
            all_results.append(risk_decision)
        if portfolio_decision:
            all_results.append(portfolio_decision)

        return all_results

    async def analyze_multiple_symbols(self, symbols: List[str]) -> Dict[str, List]:
        """Analyze multiple symbols concurrently"""
        async def analyze_symbol(symbol: str):
            # Get market data for symbol
            market_data = await self.data_pipeline.get_market_data(symbol)
            # Run analysis
            return await self.parallel_analysis(symbol, market_data)

        # Create tasks for all symbols
        symbol_tasks = [analyze_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*symbol_tasks)

        return dict(zip(symbols, results))
