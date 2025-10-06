# Testing Structure Migration Guide

This guide provides step-by-step instructions for migrating from the current scattered testing structure to the new organized testing framework.

## Overview

The migration involves:
1. Creating the new directory structure
2. Moving and refactoring existing test files
3. Updating import statements
4. Configuring the new testing framework
5. Updating CI/CD pipelines

## Pre-Migration Checklist

- [ ] Backup current test files
- [ ] Ensure all current tests are passing
- [ ] Document any custom test configurations
- [ ] Identify any test dependencies that might break
- [ ] Schedule migration during a low-activity period

## Migration Steps

### Phase 1: Setup New Structure

#### 1.1 Create Directory Structure

```bash
# Create new test directories
mkdir -p tests/{unit/{agents,data/indicators,signal_generation/components,communication,llm,api},integration,e2e,performance,validation,comparison,fixtures/{market_data,mock_responses,test_configs}}

# Create new script directories
mkdir -p scripts/{testing,profiling,migration}

# Create documentation directories
mkdir -p docs/{testing,architecture,examples/{test_examples,testing_recipes}}
```

#### 1.2 Create Essential Configuration Files

The following files have already been created:
- `tests/conftest.py` - Shared fixtures and configuration
- `tests/__init__.py` - Test package initialization
- `tests/run_all_tests.py` - Comprehensive test runner
- `docs/testing/README.md` - Testing documentation

### Phase 2: Migrate Unit Tests

#### 2.1 Move Indicator Tests

```bash
# Move indicator tests to new location
mv tests/unit/test_trend_indicators.py tests/unit/data/indicators/
mv tests/unit/test_momentum_indicators.py tests/unit/data/indicators/
mv tests/unit/test_volatility_indicators.py tests/unit/data/indicators/
mv tests/unit/test_volume_indicators.py tests/unit/data/indicators/
mv tests/unit/test_mean_reversion_indicators.py tests/unit/data/indicators/
mv tests/unit/test_statistical_indicators.py tests/unit/data/indicators/
```

#### 2.2 Move Agent Tests

```bash
# Move agent tests
mv tests/unit/test_technical_agent.py tests/unit/agents/
# Create additional agent test files as needed
touch tests/unit/agents/test_sentiment_agent.py
touch tests/unit/agents/test_risk_agent.py
touch tests/unit/agents/test_portfolio_agent.py
```

#### 2.3 Move Signal Generation Tests

```bash
# Move signal generation tests
mv tests/unit/signal_generation/test_core.py tests/unit/signal_generation/
# Create component test directories
mkdir -p tests/unit/signal_generation/components
```

#### 2.4 Move Communication Tests

```bash
# Move communication tests
mv tests/unit/test_state_manager.py tests/unit/communication/
# Create additional communication test files
touch tests/unit/communication/test_message_bus.py
touch tests/unit/communication/test_orchestrator.py
```

### Phase 3: Migrate Integration Tests

#### 3.1 Rename and Update Integration Tests

```bash
# Rename for clarity
mv tests/integration/test_data_pipeline.py tests/integration/test_data_pipeline_integration.py
mv tests/integration/test_orchestration.py tests/integration/test_orchestration_integration.py
mv tests/integration/test_signal_generation.py tests/integration/test_signal_generator_integration.py
```

#### 3.2 Update Import Statements

For each moved file, update imports:

```python
# Old imports
from tests.unit.test_technical_agent import TestTechnicalAgent

# New imports
from tests.unit.agents.test_technical_agent import TestTechnicalAgent
```

### Phase 4: Migrate Root Directory Tests

#### 4.1 Move End-to-End Tests

```bash
# Move root directory tests to e2e
mv test_signal_generation.py tests/e2e/test_signal_generation_e2e.py
mv test_signal_generation_simple.py tests/e2e/test_signal_generation_e2e_simple.py
```
✅ **COMPLETED**: Files have been moved to `tests/e2e/` directory

#### 4.2 Convert Demo Scripts to Test Utilities

```bash
# Move demo scripts to fixtures for reuse
mv test_statistical_demo.py tests/fixtures/generate_statistical_data.py
mv test_volume_demo.py tests/fixtures/generate_volume_data.py
```
✅ **COMPLETED**: Files have been moved to `tests/unit/data/indicators/` directory

### Phase 5: Organize Scripts

#### 5.1 Move Profiling Scripts

```bash
# Move profiling scripts
mv scripts/profile_agents.py scripts/profiling/
mv scripts/profile_data_pipeline.py scripts/profiling/
mv scripts/profile_llm_client.py scripts/profiling/
mv scripts/profile_orchestrator.py scripts/profiling/
mv scripts/performance_dashboard.py scripts/profiling/
```

#### 5.2 Move Migration Scripts

```bash
# Move migration and testing scripts
mv scripts/rollback_procedure_test.py scripts/migration/
mv scripts/side_by_side_comparison.py tests/comparison/test_local_vs_llm.py
```
✅ **COMPLETED**: These specialized testing scripts have been removed as they are no longer needed with the new testing structure

### Phase 6: Update Configuration

#### 6.1 Update pyproject.toml

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
    "--tb=short"
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "e2e: End-to-end tests",
    "performance: Performance tests",
    "validation: Validation tests",
    "comparison: Comparison tests",
    "slow: Tests that take longer to run"
]
```

#### 6.2 Update Test Scripts

Update any scripts that reference old test paths:

```python
# Old script references
subprocess.run(["python", "-m", "pytest", "tests/unit/test_technical_agent.py"])

# New script references
subprocess.run(["uv", "run", "pytest", "tests/unit/agents/test_technical_agent.py"])
```

### Phase 7: Update CI/CD

#### 7.1 Update GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install uv
        run: pip install uv
      - name: Install dependencies
        run: uv sync --dev
      - name: Run tests
        run: uv run tests/run_all_tests.py --category all
```

## Post-Migration Validation

### 1. Run Test Suite

```bash
# Run all tests to ensure nothing is broken
uv run tests/run_all_tests.py --category all

# Run with coverage to check coverage levels
uv run tests/run_all_tests.py --category unit --coverage
```

### 2. Check Import Paths

Verify all imports are working correctly:

```bash
# Check for import errors
uv run pytest --collect-only
```

### 3. Validate Test Categories

Ensure tests are properly categorized:

```bash
# Check that all tests have appropriate markers
uv run pytest --collect-only | grep -E "(unit|integration|e2e|performance|validation|comparison)"
```

### 4. Update Documentation

Update any documentation that references old test structure:

- README.md
- Developer guides
- API documentation
- Architecture documents

## Common Migration Issues and Solutions

### Issue 1: Import Path Errors

**Problem**: Tests can't find modules after moving
**Solution**: Update import statements and ensure `__init__.py` files exist

```python
# Add __init__.py files to new directories
touch tests/unit/agents/__init__.py
touch tests/unit/data/__init__.py
touch tests/unit/data/indicators/__init__.py
```

### Issue 2: Fixture Not Found

**Problem**: Tests can't find fixtures after migration
**Solution**: Ensure fixtures are in `conftest.py` or appropriate fixture files

```python
# Check fixture visibility
uv run pytest --collect-only | grep fixture
```

### Issue 3: Test Markers Not Recognized

**Problem**: Pytest doesn't recognize new test markers
**Solution**: Update `pyproject.toml` configuration

```toml
[tool.pytest.ini_options]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    # ... add all markers
]
```

### Issue 4: Circular Imports

**Problem**: New structure creates circular imports
**Solution**: Reorganize imports and use dependency injection

```python
# Use relative imports within test packages
from ..fixtures import sample_market_data
```

## Rollback Plan

If migration causes issues:

1. **Immediate Rollback**: Restore from Git backup
2. **Partial Rollback**: Move problematic files back to original locations
3. **Gradual Migration**: Migrate one category at a time

### Rollback Commands

```bash
# Rollback all changes
git checkout HEAD~1 -- tests/ scripts/

# Rollback specific category
git checkout HEAD~1 -- tests/unit/
```

## Migration Timeline

| Day | Task | Status |
|-----|------|--------|
| 1 | Create new directory structure | |
| 1 | Create configuration files | |
| 2 | Migrate unit tests | |
| 3 | Migrate integration tests | |
| 3 | Migrate e2e tests | |
| 4 | Organize scripts | |
| 4 | Update configuration | |
| 5 | Update CI/CD | |
| 5 | Validate migration | |
| 6 | Documentation updates | |
| 6 | Cleanup old files | |

## Support and Resources

- **Testing Documentation**: `docs/testing/README.md`
- **Test Runner**: `tests/run_all_tests.py`
- **Shared Fixtures**: `tests/conftest.py`
- **Team Communication**: Create a dedicated channel for migration questions

## Best Practices for Migration

1. **Test in Small Batches**: Migrate one category at a time
2. **Keep Backups**: Use Git branches for each migration phase
3. **Test Continuously**: Run tests after each migration step
4. **Document Changes**: Keep track of what was changed and why
5. **Communicate**: Keep the team informed about migration progress

## Post-Migration Improvements

After migration, consider these improvements:

1. **Add Missing Tests**: Fill gaps in test coverage
2. **Improve Fixtures**: Create more reusable fixtures
3. **Add Performance Tests**: Implement performance benchmarks
4. **Enhance Documentation**: Add more examples and guides
5. **Automate More**: Add more automation to testing workflow

## Conclusion

This migration will significantly improve the organization and maintainability of your testing framework. The new structure follows industry best practices and will make it easier to:

- Find and run specific tests
- Understand test coverage
- Maintain and extend tests
- Onboard new team members

Take your time with the migration, test thoroughly at each step, and don't hesitate to ask for help if you encounter issues.
