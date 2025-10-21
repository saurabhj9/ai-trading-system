# Configuration Architecture Guide

This document explains the AI Trading System's 3-tier configuration architecture, which provides optimal separation of concerns between secrets, feature flags, and business logic configuration.

## Architecture Overview

### Tier 1: Code Defaults (`src/config/settings.py`)
- **Purpose**: Default values for all feature flags and business logic
- **Visibility**: Version controlled, visible in PRs and code reviews
- **Benefits**: Provides sensible production-ready defaults
- **Example**: `LOCAL_SIGNAL_GENERATION_ENABLED: bool = True`

### Tier 2: Config Files (Optional: `config/*.yaml`)
- **Purpose**: Environment-specific configurations (dev/staging/prod)
- **Visibility**: Version controlled, easy to review
- **Use Case**: Complex nested configurations, team-specific settings
- **Status**: Planned for future implementation

### Tier 3: Environment Variables (`.env`)
- **Purpose**: Only secrets and credentials (API keys, tokens, endpoints)
- **Visibility**: Never committed to git, kept secure
- **Override Power**: Can override any Tier 1/2 setting for local development
- **Example**: `DATA_ALPHA_VANTAGE_API_KEY="your-key-here"`

## Benefits

✅ **Clear Separation**: Secrets isolated from business logic configuration
✅ **Version Control**: All feature flag changes visible in PRs
✅ **Documentation**: Self-documenting through code defaults
✅ **Maintainability**: Easy to understand and modify defaults
✅ **Flexibility**: Local overrides still possible via environment variables
✅ **Security**: No secrets in git history
✅ **12-Factor**: Follows industry best practices

## Current Configuration Categories

### Signal Generation Settings
Located in `SignalGenerationSettings` class with production-ready defaults:

```python
# Integration mode settings
LOCAL_SIGNAL_GENERATION_ENABLED: bool = True      # Enable local rule-based signals
HYBRID_MODE_ENABLED: bool = True                  # Try local first, escalate to LLM
FALLBACK_TO_LLM_ON_ERROR: bool = True             # Use LLM as fallback

# Gradual rollout controls
ROLLOUT_PERCENTAGE: float = 1.0                   # 100% use local generation
ENABLED_SYMBOLS: List[str] = []                   # All symbols eligible
ENABLED_TIMEFRAMES: List[str] = []                # All timeframes eligible

# Performance comparison
ENABLE_SIDE_BY_SIDE_COMPARISON: bool = False       # Research mode
COMPARISON_SAMPLE_RATE: float = 0.1               # When comparison enabled

# Escalation settings
ESCALATION_CONFIDENCE_THRESHOLD: float = 0.3      # Min confidence to avoid escalation
```

### LLM Configuration
Located in `LLMSettings` class:

```python
PROVIDER: str = "openrouter"                      # Default provider
DEFAULT_MODEL: str = "anthropic/claude-3-haiku"   # Default model
CACHE_TTL_SECONDS: int = 3600                     # 1 hour caching
```

### Data Provider Settings
Located in `DataSettings` class:

```python
CACHE_ENABLED: bool = True                        # Enable caching
ALPHA_VANTAGE_API_KEY: Optional[str] = None       # Set via .env
MARKETAUX_API_KEY: Optional[str] = None           # Set via .env
FINNHUB_API_KEY: Optional[str] = None             # Set via .env
```

## Override Patterns

### Example 1: Disable Local Signal Generation
**Environment Variable:**
```bash
SIGNAL_GENERATION_LOCAL_SIGNAL_GENERATION_ENABLED=false
```

**Effect:** System will use only LLM-based analysis, ignoring the default `True` setting.

### Example 2: Gradual Rollout (50%)
**Environment Variable:**
```bash
SIGNAL_GENERATION_ROLLOUT_PERCENTAGE=0.5
```

**Effect:** 50% of requests will use local generation, 50% will use LLM.

### Example 3: Symbol-Specific Testing
**Environment Variable:**
```bash
SIGNAL_GENERATION_ENABLED_SYMBOLS=["AAPL", "MSFT", "GOOGL"]
```

**Effect:** Only AAPL, MSFT, and GOOGL will use local generation; other symbols use LLM.

### Example 4: Stricter Quality Control
**Environment Variable:**
```bash
SIGNAL_GENERATION_ESCALATION_CONFIDENCE_THRESHOLD=0.7
```

**Effect:** Local signals need ≥70% confidence, otherwise escalate to LLM (default is 30%).

## Environment Variables vs Code Defaults

| Setting | Code Default | Environment Override | Final Value |
|---------|--------------|---------------------|-------------|
| LOCAL_SIGNAL_GENERATION_ENABLED | True | SIGNAL_GENERATION_LOCAL_SIGNAL_GENERATION_ENABLED=false | False |
| HYBRID_MODE_ENABLED | True | (not set) | True |
| ESCALATION_CONFIDENCE_THRESHOLD | 0.3 | SIGNAL_GENERATION_ESCALATION_CONFIDENCE_THRESHOLD=0.5 | 0.5 |

## Migration Guide

### For Existing Deployments

1. **No Breaking Changes**: All existing environment variables continue to work
2. **New Defaults**: Production-ready defaults are now available in code
3. **Optional Cleanup**: Can remove feature flags from `.env` to use code defaults

### Recommended `.env` Structure

**Good (Current Approach):**
```bash
# Only secrets and credentials
DATA_ALPHA_VANTAGE_API_KEY="your-key-here"
DATA_FINNHUB_API_KEY="your-key-here"
OPENROUTER_API_KEY="your-key-here"

# Optional overrides for local development
# SIGNAL_GENERATION_LOCAL_SIGNAL_GENERATION_ENABLED=false
```

**Avoid (Old Approach):**
```bash
# Don't mix feature flags with secrets
SIGNAL_GENERATION_LOCAL_SIGNAL_GENERATION_ENABLED=true
SIGNAL_GENERATION_ROLLOUT_PERCENTAGE=1.0
DATA_ALPHA_VANTAGE_API_KEY="your-key-here"
```

## Best Practices

### Adding New Configuration

1. **Add Default to Settings:**
```python
class NewFeatureSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='NEW_FEATURE_')

    FEATURE_ENABLED: bool = True  # Sensible default
    THRESHOLD: float = 0.5        # Documented default
```

2. **Update `.env.example` if Secret:**
```bash
NEW_FEATURE_API_KEY="YOUR-API-KEY-HERE"
```

3. **Document Override Pattern:**
```bash
# NEW_FEATURE_FEATURE_ENABLED=false  # Optional override
```

### Security Guidelines

- ✅ API keys, tokens, passwords → `.env` only
- ✅ Feature flags, thresholds → `settings.py` defaults
- ✅ Local development overrides → `.env`
- ❌ Never commit secrets to git
- ❌ Don't put feature flags in `.env` unless overriding

### Code Review Guidelines

- ✅ Review feature flag changes in PRs (visible in settings.py)
- ✅ Ensure no secrets in code commits
- ✅ Check new settings have sensible defaults
- ✅ Verify environment variable naming consistency

## Troubleshooting

### Common Issues

1. **Configuration Not Loading:**
   - Check pydantic-settings import
   - Verify `.env` file exists and is readable
   - Check for syntax errors in settings.py

2. **Environment Override Not Working:**
   - Verify variable name uses correct prefix (e.g., `SIGNAL_GENERATION_`)
   - Check for typos in variable name
   - Ensure environment is actually loaded

3. **Type Validation Errors:**
   - Check variable type matches expected (bool, float, List[str])
   - List format: `["AAPL", "MSFT"]` not `"AAPL,MSFT"`

### Debug Commands

```python
# Test configuration loading
from src.config.settings import settings
print(settings.signal_generation.LOCAL_SIGNAL_GENERATION_ENABLED)

# Check what's actually set
import os
print(os.getenv('SIGNAL_GENERATION_LOCAL_SIGNAL_GENERATION_ENABLED'))
```

## Future Enhancements

### Tier 2: YAML Configuration
Planned support for environment-specific YAML configs:

```yaml
# config/production.yaml
signal_generation:
  local_signal_generation_enabled: true
  rollout_percentage: 1.0
  escalation_confidence_threshold: 0.3

# config/development.yaml
signal_generation:
  local_signal_generation_enabled: false
  enable_side_by_side_comparison: true
```

### Configuration Validation
Enhanced validation and schema documentation in future updates.

## Files Reference

- **`src/config/settings.py`** - Main configuration definitions
- **`.env.example`** - Secrets template and override examples
- **`docs/CONFIGURATION.md`** - This documentation
- **`docs/TODO.md`** - Current task progress
