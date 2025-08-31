# Changelog

All notable changes to Modern ML Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-08-31

### 🚨 BREAKING CHANGES

#### Required Parameters
- `--env-name` is now **required** for all execution commands
- No fallback to environment variables for env_name parameter

#### Directory Structure
- Only `configs/` directory is supported (not `config/`)
- Only `recipes/` directory is supported (not `models/recipes/`)

#### Removed Features
- `environment.app_env` field removed (use `environment.env_name` property instead)
- Single `.env` file not supported (use `.env.{env_name}` files)
- Config merging from APP_ENV removed
- `get_env_name_with_fallback()` function removed
- `migrate` command removed
- Deprecation utility module removed

### ✨ Added
- `environment.env_name` property that reads from ENV_NAME environment variable
- Clean v2.0 API without legacy compatibility layers
- Cleanup scripts for removing legacy code:
  - `scripts/cleanup_legacy.sh` - Remove legacy files and directories
  - `scripts/scan_legacy_code.py` - Scan for remaining legacy patterns

### 🔧 Changed
- `load_config_files()` now requires `env_name` parameter
- `load_settings_by_file()` requires `env_name` parameter  
- All CLI commands require explicit `--env-name` parameter
- Settings module architecture simplified without backward compatibility

### 🗑️ Removed
- Legacy Settings loading functions
- Backward compatibility layers  
- Deprecation warnings (code is now removed)
- TYPE_CHECKING circular dependencies
- Support for `config/` directory
- Support for `models/recipes/` directory
- Support for single `.env` file

### 📈 Improved
- Cleaner codebase without legacy code
- Better performance without compatibility checks
- Simplified Settings module architecture
- Clear separation of Recipe and Config

### 📚 Documentation
- Updated README with v2.0 usage examples
- Migration guide for upgrading from v1.x
- Complete API documentation for v2.0

## [1.10.0] - 2024-XX-XX (Transition Release)

### Added
- Migration tool for upgrading to v2.0
- Deprecation warnings for all features removed in v2.0

### Changed
- Final release supporting both old and new patterns
- Includes `mmp migrate` command for automated migration

## Migration Guide

### From v1.x to v2.0

If you're upgrading from v1.x, follow these steps:

1. **Install transition version first:**
   ```bash
   pip install modern-ml-pipeline==1.10
   ```

2. **Run migration tool:**
   ```bash
   mmp migrate
   ```

3. **Update to v2.0:**
   ```bash
   pip install --upgrade modern-ml-pipeline
   ```

4. **Update your code:**
   - Replace `environment.app_env` with `environment.env_name`
   - Add `--env-name` to all CLI commands
   - Rename `config/` to `configs/`
   - Use `.env.{env_name}` instead of `.env`

### Key Changes to Note

1. **CLI Commands now require env_name:**
   ```bash
   # Old (v1.x)
   mmp train recipes/model.yaml
   
   # New (v2.0)
   mmp train --recipe-file recipes/model.yaml --env-name dev
   ```

2. **Settings access changed:**
   ```python
   # Old (v1.x)
   env = settings.environment.app_env
   
   # New (v2.0)
   env = settings.environment.env_name
   ```

3. **Directory structure:**
   ```
   # Old (v1.x)
   config/
   ├── base.yaml
   └── dev.yaml
   
   # New (v2.0)
   configs/
   ├── base.yaml
   └── dev.yaml
   ```

## Support

For issues or questions about upgrading to v2.0:
- Open an issue on GitHub
- Check the migration guide in docs/
- Review example projects in examples/