"""
Migration utilities for transitioning from LLM-only to hybrid signal generation.

This module provides tools and utilities to help with the gradual migration
from LLM-only signal generation to the hybrid approach using LocalSignalGenerator.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from ..config.settings import settings
from ..agents.technical import TechnicalAnalysisAgent
from ..communication.orchestrator import Orchestrator
from ..data.pipeline import DataPipeline


class SignalGenerationMigration:
    """
    Manages the migration from LLM-only to hybrid signal generation.

    This class provides utilities for:
    - Gradual rollout of local signal generation
    - Performance comparison between local and LLM signals
    - Configuration management for migration phases
    - Rollback procedures if issues arise
    """

    def __init__(self):
        self.migration_config = self._load_migration_config()
        self.migration_log = []
        self.performance_history = []

    def _load_migration_config(self) -> Dict[str, Any]:
        """Load migration configuration from file or create default."""
        config_path = Path("config/migration_config.json")

        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)

        # Default migration configuration
        default_config = {
            "migration_phases": [
                {
                    "name": "shadow_mode",
                    "description": "Run local signals in parallel with LLM without executing trades",
                    "duration_days": 7,
                    "rollout_percentage": 0.0,
                    "enable_comparison": True,
                    "comparison_sample_rate": 1.0,
                    "hybrid_mode": False,
                    "escalation_enabled": False,
                },
                {
                    "name": "limited_rollout",
                    "description": "Enable local signals for a small subset of assets",
                    "duration_days": 7,
                    "rollout_percentage": 0.1,
                    "enable_comparison": True,
                    "comparison_sample_rate": 0.5,
                    "hybrid_mode": False,
                    "escalation_enabled": True,
                    "escalation_confidence_threshold": 0.3,
                },
                {
                    "name": "gradual_expansion",
                    "description": "Gradually increase rollout percentage",
                    "duration_days": 14,
                    "rollout_percentage": 0.5,
                    "enable_comparison": True,
                    "comparison_sample_rate": 0.2,
                    "hybrid_mode": True,
                    "escalation_enabled": True,
                    "escalation_confidence_threshold": 0.4,
                },
                {
                    "name": "full_migration",
                    "description": "Complete migration to hybrid approach",
                    "duration_days": 7,
                    "rollout_percentage": 1.0,
                    "enable_comparison": False,
                    "comparison_sample_rate": 0.0,
                    "hybrid_mode": True,
                    "escalation_enabled": True,
                    "escalation_confidence_threshold": 0.5,
                }
            ],
            "current_phase": 0,
            "migration_start_date": None,
            "phase_start_date": None,
            "rollback_triggers": {
                "error_rate_threshold": 5.0,  # percentage
                "performance_degradation_threshold": 20.0,  # percentage
                "min_confidence_threshold": 0.6,
            },
            "success_metrics": {
                "target_performance_improvement": 50.0,  # percentage
                "target_cost_reduction": 70.0,  # percentage
                "min_signal_accuracy": 0.8,
            }
        }

        # Create config directory if it doesn't exist
        config_path.parent.mkdir(exist_ok=True)

        # Save default configuration
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

        return default_config

    def get_current_phase(self) -> Dict[str, Any]:
        """Get the current migration phase configuration."""
        phase_index = self.migration_config.get("current_phase", 0)
        phases = self.migration_config.get("migration_phases", [])

        if phase_index >= len(phases):
            return phases[-1]  # Return last phase if index is out of bounds

        return phases[phase_index]

    def advance_to_next_phase(self) -> bool:
        """
        Advance to the next migration phase.

        Returns:
            bool: True if advanced successfully, False if already at last phase
        """
        current_phase = self.migration_config.get("current_phase", 0)
        total_phases = len(self.migration_config.get("migration_phases", []))

        if current_phase >= total_phases - 1:
            return False  # Already at last phase

        self.migration_config["current_phase"] = current_phase + 1
        self.migration_config["phase_start_date"] = datetime.now().isoformat()
        self._save_migration_config()

        self._log_migration_event(
            "phase_advanced",
            f"Advanced to phase {current_phase + 1}: {self.get_current_phase()['name']}"
        )

        return True

    def apply_phase_configuration(self) -> Dict[str, Any]:
        """
        Apply the current migration phase configuration to settings.

        Returns:
            Dict[str, Any]: The applied configuration
        """
        phase = self.get_current_phase()

        # Create environment file updates
        env_updates = {
            "SIGNAL_GENERATION_LOCAL_SIGNAL_GENERATION_ENABLED": "true",
            "SIGNAL_GENERATION_ROLLOUT_PERCENTAGE": str(phase["rollout_percentage"]),
            "SIGNAL_GENERATION_ENABLE_SIDE_BY_SIDE_COMPARISON": str(phase["enable_comparison"]).lower(),
            "SIGNAL_GENERATION_COMPARISON_SAMPLE_RATE": str(phase["comparison_sample_rate"]),
            "SIGNAL_GENERATION_HYBRID_MODE_ENABLED": str(phase["hybrid_mode"]).lower(),
            "SIGNAL_GENERATION_ESCALATION_ENABLED": str(phase["escalation_enabled"]).lower(),
        }

        # Add escalation threshold if specified
        if "escalation_confidence_threshold" in phase:
            env_updates["SIGNAL_GENERATION_ESCALATION_CONFIDENCE_THRESHOLD"] = str(
                phase["escalation_confidence_threshold"]
            )

        # Update .env file
        self._update_env_file(env_updates)

        self._log_migration_event(
            "configuration_applied",
            f"Applied configuration for phase: {phase['name']}",
            {"configuration": env_updates}
        )

        return env_updates

    def check_rollback_conditions(self, performance_metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if rollback conditions are met based on performance metrics.

        Args:
            performance_metrics: Current performance metrics

        Returns:
            Tuple[bool, List[str]]: (should_rollback, list_of_reasons)
        """
        rollback_triggers = self.migration_config.get("rollback_triggers", {})
        reasons = []

        # Check error rate
        error_rate = performance_metrics.get("local_error_rate", 0.0)
        if error_rate > rollback_triggers.get("error_rate_threshold", 5.0):
            reasons.append(f"High error rate: {error_rate:.2f}% > {rollback_triggers['error_rate_threshold']}%")

        # Check performance degradation
        performance_improvement = performance_metrics.get("performance_improvement", 0.0)
        if performance_improvement < -rollback_triggers.get("performance_degradation_threshold", 20.0):
            reasons.append(
                f"Performance degradation: {performance_improvement:.2f}% < "
                f"-{rollback_triggers['performance_degradation_threshold']}%"
            )

        # Check minimum confidence
        avg_confidence = performance_metrics.get("avg_validation_confidence", 0.0)
        if avg_confidence < rollback_triggers.get("min_confidence_threshold", 0.6):
            reasons.append(
                f"Low confidence: {avg_confidence:.2f} < {rollback_triggers['min_confidence_threshold']}"
            )

        return len(reasons) > 0, reasons

    def execute_rollback(self) -> bool:
        """
        Execute rollback to LLM-only signal generation.

        Returns:
            bool: True if rollback was successful
        """
        try:
            # Disable local signal generation
            env_updates = {
                "SIGNAL_GENERATION_LOCAL_SIGNAL_GENERATION_ENABLED": "false",
                "SIGNAL_GENERATION_HYBRID_MODE_ENABLED": "false",
                "SIGNAL_GENERATION_ESCALATION_ENABLED": "false",
            }

            self._update_env_file(env_updates)

            # Reset migration phase to shadow mode
            self.migration_config["current_phase"] = 0
            self._save_migration_config()

            self._log_migration_event(
                "rollback_executed",
                "Rollback to LLM-only signal generation completed",
                {"configuration": env_updates}
            )

            return True

        except Exception as e:
            self._log_migration_event(
                "rollback_failed",
                f"Rollback failed: {str(e)}"
            )
            return False

    def check_migration_success(self, performance_metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if migration success criteria are met.

        Args:
            performance_metrics: Current performance metrics

        Returns:
            Tuple[bool, List[str]]: (is_successful, list_of_unmet_criteria)
        """
        success_metrics = self.migration_config.get("success_metrics", {})
        unmet_criteria = []

        # Check performance improvement
        performance_improvement = performance_metrics.get("performance_improvement", 0.0)
        if performance_improvement < success_metrics.get("target_performance_improvement", 50.0):
            unmet_criteria.append(
                f"Performance improvement: {performance_improvement:.2f}% < "
                f"{success_metrics['target_performance_improvement']}%"
            )

        # Check cost reduction (would need to be calculated from actual costs)
        # This is a placeholder - actual cost calculation would depend on your billing system
        cost_reduction = performance_metrics.get("cost_reduction", 0.0)
        if cost_reduction < success_metrics.get("target_cost_reduction", 70.0):
            unmet_criteria.append(
                f"Cost reduction: {cost_reduction:.2f}% < {success_metrics['target_cost_reduction']}%"
            )

        # Check signal accuracy
        signal_accuracy = performance_metrics.get("signal_accuracy", 0.0)
        if signal_accuracy < success_metrics.get("min_signal_accuracy", 0.8):
            unmet_criteria.append(
                f"Signal accuracy: {signal_accuracy:.2f} < {success_metrics['min_signal_accuracy']}"
            )

        return len(unmet_criteria) == 0, unmet_criteria

    def _update_env_file(self, updates: Dict[str, str]):
        """Update the .env file with new configuration."""
        env_path = Path(".env")

        # Read existing .env file
        env_vars = {}
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key] = value

        # Apply updates
        env_vars.update(updates)

        # Write back to .env file
        with open(env_path, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

    def _save_migration_config(self):
        """Save migration configuration to file."""
        config_path = Path("config/migration_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.migration_config, f, indent=2)

    def _log_migration_event(self, event_type: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log a migration event."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "data": data or {},
        }

        self.migration_log.append(log_entry)

        # Also write to log file
        log_path = Path("logs/migration.log")
        log_path.parent.mkdir(exist_ok=True)

        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_migration_status(self) -> Dict[str, Any]:
        """
        Get the current migration status.

        Returns:
            Dict[str, Any]: Migration status information
        """
        current_phase = self.get_current_phase()
        phase_index = self.migration_config.get("current_phase", 0)
        total_phases = len(self.migration_config.get("migration_phases", []))

        return {
            "current_phase": phase_index,
            "total_phases": total_phases,
            "phase_name": current_phase["name"],
            "phase_description": current_phase["description"],
            "migration_progress": (phase_index + 1) / total_phases * 100,
            "recent_events": self.migration_log[-10:] if self.migration_log else [],
            "configuration": current_phase,
        }


# Convenience functions for migration management
def initialize_migration() -> SignalGenerationMigration:
    """Initialize and return a migration manager."""
    return SignalGenerationMigration()


def apply_migration_phase() -> Dict[str, Any]:
    """Apply the current migration phase configuration."""
    migration = initialize_migration()
    return migration.apply_phase_configuration()


def check_and_handle_rollback(performance_metrics: Dict[str, Any]) -> bool:
    """
    Check rollback conditions and execute rollback if needed.

    Args:
        performance_metrics: Current performance metrics

    Returns:
        bool: True if rollback was executed
    """
    migration = initialize_migration()
    should_rollback, reasons = migration.check_rollback_conditions(performance_metrics)

    if should_rollback:
        print(f"Rollback triggered: {', '.join(reasons)}")
        return migration.execute_rollback()

    return False


def advance_migration_phase() -> bool:
    """Advance to the next migration phase."""
    migration = initialize_migration()
    success = migration.advance_to_next_phase()

    if success:
        migration.apply_phase_configuration()

    return success
