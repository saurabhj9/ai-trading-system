"""
Configuration management for the event-driven trigger system.

This module provides centralized configuration management for all trigger system
components including polling, detection settings, cooldown periods, and
integration parameters.
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

from .trigger_detector import TriggerConfig
from ..data.real_time_poller import PollingConfig, SymbolPriority
from .cooldown_manager import CooldownConfig
from .trigger_integration import TriggerSystemConfig


@dataclass
class TriggerEnvironmentConfig:
    """Environment-specific trigger configuration."""

    # Data polling settings
    polling_enabled: bool = False
    polling_high_priority_interval_seconds: int = 5
    polling_medium_priority_interval_seconds: int = 15
    polling_low_priority_interval_seconds: int = 60

    # Market hours
    market_hours_filter_enabled: bool = True
    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0

    # Technical triggers
    technical_triggers_enabled: bool = False
    technical_min_confidence: float = 0.7
    technical_cooldown_seconds: int = 300

    # Event processing
    event_bus_workers: int = 5
    max_events_per_minute: int = 50
    event_queue_size: int = 1000

    # Integration
    auto_signal_generation: bool = False
    max_concurrent_analyses: int = 3
    analysis_timeout_seconds: int = 30

    # Decision caching
    decision_ttl_default_seconds: int = 1800
    decision_ttl_critical_seconds: int = 900

    # Performance
    enable_real_time_cache: bool = True
    cache_ttl_seconds: int = 5

    @classmethod
    def from_environment(cls) -> 'TriggerEnvironmentConfig':
        """Create config from environment variables."""
        return cls(
            polling_enabled=os.getenv('TRIGGER_POLLING_ENABLED', 'false').lower() == 'true',
            polling_high_priority_interval_seconds=int(os.getenv('TRIGGER_HIGH_PRIORITY_INTERVAL', '5')),
            polling_medium_priority_interval_seconds=int(os.getenv('TRIGGER_MEDIUM_PRIORITY_INTERVAL', '15')),
            polling_low_priority_interval_seconds=int(os.getenv('TRIGGER_LOW_PRIORITY_INTERVAL', '60')),

            market_hours_filter_enabled=os.getenv('TRIGGER_MARKET_HOURS_FILTER', 'true').lower() == 'true',
            market_open_hour=int(os.getenv('TRIGGER_MARKET_OPEN_HOUR', '9')),
            market_open_minute=int(os.getenv('TRIGGER_MARKET_OPEN_MINUTE', '30')),
            market_close_hour=int(os.getenv('TRIGGER_MARKET_CLOSE_HOUR', '16')),
            market_close_minute=int(os.getenv('TRIGGER_MARKET_CLOSE_MINUTE', '0')),

            technical_triggers_enabled=os.getenv('TRIGGER_TECHNICAL_ENABLED', 'false').lower() == 'true',
            technical_min_confidence=float(os.getenv('TRIGGER_TECHNICAL_MIN_CONFIDENCE', '0.7')),
            technical_cooldown_seconds=int(os.getenv('TRIGGER_TECHNICAL_COOLDOWN_SECONDS', '300')),

            event_bus_workers=int(os.getenv('TRIGGER_EVENT_BUS_WORKERS', '5')),
            max_events_per_minute=int(os.getenv('TRIGGER_MAX_EVENTS_PER_MINUTE', '50')),
            event_queue_size=int(os.getenv('TRIGGER_EVENT_QUEUE_SIZE', '1000')),

            auto_signal_generation=os.getenv('TRIGGER_AUTO_SIGNAL_GENERATION', 'false').lower() == 'true',
            max_concurrent_analyses=int(os.getenv('TRIGGER_MAX_CONCURRENT_ANALYSES', '3')),
            analysis_timeout_seconds=int(os.getenv('TRIGGER_ANALYSIS_TIMEOUT_SECONDS', '30')),

            decision_ttl_default_seconds=int(os.getenv('TRIGGER_DECISION_TTL_DEFAULT', '1800')),
            decision_ttl_critical_seconds=int(os.getenv('TRIGGER_DECISION_TTL_CRITICAL', '900')),

            enable_real_time_cache=os.getenv('TRIGGER_ENABLE_REAL_TIME_CACHE', 'true').lower() == 'true',
            cache_ttl_seconds=int(os.getenv('TRIGGER_CACHE_TTL_SECONDS', '5'))
        )


def create_polling_config(env_config: TriggerEnvironmentConfig) -> PollingConfig:
    """Create polling configuration from environment config."""
    from datetime import time

    return PollingConfig(
        high_priority_interval=env_config.polling_high_priority_interval_seconds,
        medium_priority_interval=env_config.polling_medium_priority_interval_seconds,
        low_priority_interval=env_config.polling_low_priority_interval_seconds,
        market_open_time=time(hour=env_config.market_open_hour, minute=env_config.market_open_minute),
        market_close_time=time(hour=env_config.market_close_hour, minute=env_config.market_close_minute),
        enable_market_hours_filter=env_config.market_hours_filter_enabled,
        enable_real_time_cache=env_config.enable_real_time_cache,
        real_time_cache_ttl=env_config.cache_ttl_seconds
    )


def create_technical_trigger_config(env_config: TriggerEnvironmentConfig) -> TriggerConfig:
    """Create technical trigger configuration from environment config."""
    return TriggerConfig(
        enabled=env_config.technical_triggers_enabled,
        min_confidence=env_config.technical_min_confidence,
        cooldown_seconds=env_config.technical_cooldown_seconds,
        max_triggers_per_minute=min(env_config.max_events_per_minute // 2, 10)
    )


def create_cooldown_config(env_config: TriggerEnvironmentConfig) -> CooldownConfig:
    """Create cooldown configuration from environment config."""
    from .cooldown_manager import TriggerSeverity

    return CooldownConfig(
        enabled=True,
        symbol_cooldown_seconds=env_config.technical_cooldown_seconds,
        trigger_type_cooldown_seconds=env_config.technical_cooldown_seconds,
        severity_cooldown_seconds={
            TriggerSeverity.LOW: env_config.decision_ttl_default_seconds,
            TriggerSeverity.MEDIUM: env_config.technical_cooldown_seconds,
            TriggerSeverity.HIGH: env_config.technical_cooldown_seconds // 2,
            TriggerSeverity.CRITICAL: env_config.decision_ttl_critical_seconds
        },
        default_decision_ttl_seconds=env_config.decision_ttl_default_seconds,
        critical_decision_ttl_seconds=env_config.decision_ttl_critical_seconds,
        enable_cache_persistence=True
    )


def create_trigger_system_config(env_config: TriggerEnvironmentConfig) -> TriggerSystemConfig:
    """Create complete trigger system configuration from environment config."""
    polling_config = create_polling_config(env_config)
    cooldown_config = create_cooldown_config(env_config)

    return TriggerSystemConfig(
        polling_config=polling_config,
        technical_enabled=env_config.technical_triggers_enabled,
        volatility_enabled=False,  # To be implemented
        trend_enabled=False,       # To be implemented
        conflict_enabled=False,    # To be implemented
        regime_enabled=False,      # To be implemented
        cooldown_config=cooldown_config,
        auto_signal_generation=env_config.auto_signal_generation,
        cache_decisions=True,
        enable_logging=True,
        max_concurrent_analyses=env_config.max_concurrent_analyses,
        analysis_timeout_seconds=env_config.analysis_timeout_seconds
    )


# Default preset configurations
class TriggerPresets:
    """Predefined trigger system configurations for different use cases."""

    @staticmethod
    def development() -> TriggerSystemConfig:
        """Configuration for development environment."""
        env_config = TriggerEnvironmentConfig(
            polling_enabled=True,
            polling_high_priority_interval_seconds=10,
            polling_medium_priority_interval_seconds=30,
            polling_low_priority_interval_seconds=120,
            technical_triggers_enabled=True,
            technical_min_confidence=0.6,
            technical_cooldown_seconds=180,
            event_bus_workers=2,
            max_events_per_minute=20,
            auto_signal_generation=True,
            max_concurrent_analyses=2,
            analysis_timeout_seconds=60
        )
        return create_trigger_system_config(env_config)

    @staticmethod
    def production() -> TriggerSystemConfig:
        """Configuration for production environment."""
        env_config = TriggerEnvironmentConfig(
            polling_enabled=True,
            polling_high_priority_interval_seconds=5,
            polling_medium_priority_interval_seconds=15,
            polling_low_priority_interval_seconds=60,
            technical_triggers_enabled=True,
            technical_min_confidence=0.8,
            technical_cooldown_seconds=300,
            event_bus_workers=5,
            max_events_per_minute=100,
            auto_signal_generation=True,
            max_concurrent_analyses=5,
            analysis_timeout_seconds=30
        )
        return create_trigger_system_config(env_config)

    @staticmethod
    def high_frequency() -> TriggerSystemConfig:
        """Configuration for high-frequency trading."""
        env_config = TriggerEnvironmentConfig(
            polling_enabled=True,
            polling_high_priority_interval_seconds=2,
            polling_medium_priority_interval_seconds=5,
            polling_low_priority_interval_seconds=30,
            technical_triggers_enabled=True,
            technical_min_confidence=0.7,
            technical_cooldown_seconds=60,
            event_bus_workers=10,
            max_events_per_minute=500,
            auto_signal_generation=True,
            max_concurrent_analyses=10,
            analysis_timeout_seconds=15
        )
        return create_trigger_system_config(env_config)

    @staticmethod
    def conservative() -> TriggerSystemConfig:
        """Configuration for conservative trading."""
        env_config = TriggerEnvironmentConfig(
            polling_enabled=True,
            polling_high_priority_interval_seconds=30,
            polling_medium_priority_interval_seconds=60,
            polling_low_priority_interval_seconds=300,
            technical_triggers_enabled=True,
            technical_min_confidence=0.9,
            technical_cooldown_seconds=900,
            event_bus_workers=2,
            max_events_per_minute=10,
            auto_signal_generation=True,
            max_concurrent_analyses=1,
            analysis_timeout_seconds=45
        )
        return create_trigger_system_config(env_config)


# Configuration loader
def load_trigger_config(
    preset: Optional[str] = None,
    config_file: Optional[Path] = None,
    use_environment: bool = True
) -> TriggerSystemConfig:
    """
    Load trigger system configuration from various sources.

    Args:
        preset: Name of preset configuration ('development', 'production', 'high_frequency', 'conservative')
        config_file: Path to configuration file (JSON/YAML support)
        use_environment: Whether to override with environment variables

    Returns:
        Complete trigger system configuration
    """
    # Start with preset or default
    if preset:
        preset_map = {
            'development': TriggerPresets.development,
            'production': TriggerPresets.production,
            'high_frequency': TriggerPresets.high_frequency,
            'conservative': TriggerPresets.conservative
        }

        if preset not in preset_map:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(preset_map.keys())}")

        config = preset_map[preset]()
    else:
        # Default to production
        config = TriggerPresets.production()

    # Override with environment if enabled
    if use_environment:
        env_config = TriggerEnvironmentConfig.from_environment()
        env_override_config = create_trigger_system_config(env_config)

        # Override specific settings
        if env_config.polling_enabled:
            config.polling_config = env_override_config.polling_config

        if env_config.technical_triggers_enabled:
            config.technical_enabled = True
            config.cooldown_config = env_override_config.cooldown_config

        config.auto_signal_generation = env_config.auto_signal_generation
        config.max_concurrent_analyses = env_config.max_concurrent_analyses
        config.analysis_timeout_seconds = env_config.analysis_timeout_seconds

    # Load from config file if provided (future enhancement)
    if config_file and config_file.exists():
        # TODO: Implement JSON/YAML config file loading
        logger.warning(f"Config file loading not yet implemented: {config_file}")

    return config


def generate_environment_template() -> str:
    """Generate environment variable template for documentation."""
    template = """
# Event-Driven Triggers Configuration

# Enable/disable real-time polling for triggers
TRIGGER_POLLING_ENABLED=false

# Polling intervals in seconds
TRIGGER_HIGH_PRIORITY_INTERVAL=5
TRIGGER_MEDIUM_PRIORITY_INTERVAL=15
TRIGGER_LOW_PRIORITY_INTERVAL=60

# Market hours filtering
TRIGGER_MARKET_HOURS_FILTER=true
TRIGGER_MARKET_OPEN_HOUR=9
TRIGGER_MARKET_OPEN_MINUTE=30
TRIGGER_MARKET_CLOSE_HOUR=16
TRIGGER_MARKET_CLOSE_MINUTE=0

# Technical trigger settings
TRIGGER_TECHNICAL_ENABLED=false
TRIGGER_TECHNICAL_MIN_CONFIDENCE=0.7
TRIGGER_TECHNICAL_COOLDOWN_SECONDS=300

# Event processing
TRIGGER_EVENT_BUS_WORKERS=5
TRIGGER_MAX_EVENTS_PER_MINUTE=50
TRIGGER_EVENT_QUEUE_SIZE=1000

# Integration settings
TRIGGER_AUTO_SIGNAL_GENERATION=false
TRIGGER_MAX_CONCURRENT_ANALYSES=3
TRIGGER_ANALYSIS_TIMEOUT_SECONDS=30

# Decision caching TTL in seconds
TRIGGER_DECISION_TTL_DEFAULT=1800
TRIGGER_DECISION_TTL_CRITICAL=900

# Performance settings
TRIGGER_ENABLE_REAL_TIME_CACHE=true
TRIGGER_CACHE_TTL_SECONDS=5
""".strip()

    return template


def validate_config(config: TriggerSystemConfig) -> List[str]:
    """
    Validate trigger system configuration.

    Args:
        config: Configuration to validate

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Validate polling intervals
    if config.polling_config.high_priority_interval < 1:
        errors.append("High priority interval must be at least 1 second")

    if config.polling_config.high_priority_interval > config.polling_config.medium_priority_interval:
        errors.append("High priority interval should be <= medium priority interval")

    if config.polling_config.medium_priority_interval > config.polling_config.low_priority_interval:
        errors.append("Medium priority interval should be <= low priority interval")

    # Validate analysis settings
    if config.max_concurrent_analyses < 1:
        errors.append("Max concurrent analyses must be at least 1")

    if config.max_concurrent_analyses > 20:
        errors.append("Max concurrent analyses should not exceed 20 for stability")

    if config.analysis_timeout_seconds < 5:
        errors.append("Analysis timeout should be at least 5 seconds")

    # Validate cooldown settings
    if config.cooldown_config:
        if config.cooldown_config.symbol_cooldown_seconds < 60:
            errors.append("Symbol cooldown should be at least 60 seconds")

        if config.cooldown_config.max_triggers_per_minute < 1:
            errors.append("Max triggers per minute must be at least 1")

    return errors


# Export for easy import
__all__ = [
    'TriggerEnvironmentConfig',
    'TriggerPresets',
    'load_trigger_config',
    'create_polling_config',
    'create_technical_trigger_config',
    'create_cooldown_config',
    'create_trigger_system_config',
    'generate_environment_template',
    'validate_config'
]
