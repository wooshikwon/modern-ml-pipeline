# Deprecation Notice

## 📅 Timeline

- **v1.9** (Current): Deprecation warnings added
- **v1.10**: Legacy support in maintenance mode  
- **v2.0**: Complete removal of deprecated features

## ⚠️ Deprecated Features

### 1. `environment.app_env` Field

**Status**: Deprecated  
**Removal**: v2.0  
**Alternative**: Use `--env-name` parameter

The `app_env` field in environment settings is no longer needed. Environment selection is now handled through the `--env-name` parameter.

```yaml
# ❌ Deprecated
environment:
  app_env: "dev"  # This field will be removed

# ✅ Recommended
# Use --env-name parameter when running commands:
# mmp train --recipe-file recipe.yaml --env-name dev
```

### 2. Single `.env` File

**Status**: Deprecated  
**Removal**: v2.0  
**Alternative**: Use `.env.<env_name>` files

Environment-specific `.env` files provide better separation of concerns.

```bash
# ❌ Deprecated
.env  # Single environment file

# ✅ Recommended
.env.local
.env.dev
.env.prod
```

### 3. `config/` Directory

**Status**: Deprecated  
**Removal**: v2.0  
**Alternative**: Use `configs/` directory

The plural form better represents multiple configuration files.

```bash
# ❌ Deprecated
config/
  ├── base.yaml
  └── config.yaml

# ✅ Recommended  
configs/
  ├── local.yaml
  ├── dev.yaml
  └── prod.yaml
```

### 4. Merged Config Loading

**Status**: Deprecated  
**Removal**: v2.0  
**Alternative**: Environment-specific config loading

Loading all configs and merging them is deprecated in favor of environment-specific loading.

```python
# ❌ Deprecated
load_config_files()  # Loads and merges all configs

# ✅ Recommended
load_config_for_env("dev")  # Loads only dev config
```

### 5. Commands Without `--env-name`

**Status**: Deprecated  
**Removal**: v2.0  
**Alternative**: Always specify `--env-name`

All execution commands now require explicit environment specification.

```bash
# ❌ Deprecated
mmp train --recipe-file recipe.yaml
mmp batch-inference --run-id abc123
mmp serve-api --run-id abc123

# ✅ Recommended
mmp train --recipe-file recipe.yaml --env-name dev
mmp batch-inference --run-id abc123 --env-name prod
mmp serve-api --run-id abc123 --env-name prod
```

### 6. Python 3.7 Support

**Status**: Deprecated  
**Removal**: v2.0  
**Alternative**: Use Python 3.8+

Python 3.7 has reached end-of-life and will no longer be supported.

## 🔄 Migration Guide

### Automatic Migration

Run the migration assistant to automatically update your project:

```bash
mmp migrate
```

This command will:
- Rename `config/` to `configs/`
- Move recipes to project root
- Create environment-specific `.env` files
- Update config file structure

### Manual Migration Steps

If you prefer to migrate manually:

1. **Rename directories**:
   ```bash
   mv config configs
   mv models/recipes recipes
   ```

2. **Create environment-specific `.env` files**:
   ```bash
   mv .env .env.local
   cp .env.local .env.dev
   cp .env.local .env.prod
   ```

3. **Update config files**:
   - Remove `app_env` field from configs
   - Use environment variables for sensitive data
   - Create separate config files for each environment

4. **Update scripts**:
   - Add `--env-name` parameter to all commands
   - Update CI/CD pipelines
   - Update documentation

### Environment Variable Changes

| Old Variable | New Variable | Notes |
|-------------|--------------|-------|
| `APP_ENV` | `ENV_NAME` | Used for environment selection |
| N/A | `SHOW_DEPRECATION_WARNINGS` | Set to `false` to suppress warnings |

## 🔕 Suppressing Warnings

### Disable All Deprecation Warnings

```bash
export SHOW_DEPRECATION_WARNINGS=false
```

### Python Warning Filter

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

## 📊 Impact Assessment

### Low Impact
- Directory renaming (`config/` → `configs/`)
- File renaming (`.env` → `.env.<env>`)

### Medium Impact
- Adding `--env-name` to commands
- Updating CI/CD scripts

### High Impact
- Removing `app_env` field
- Changing config loading logic

## 💬 Getting Help

If you need assistance with migration:

1. Run `mmp migrate --dry-run` to preview changes
2. Check the [Migration Guide](docs/MIGRATION_GUIDE.md)
3. Open an issue on GitHub
4. Contact the development team

## 🚀 Benefits of Migration

After migrating to the new structure, you'll benefit from:

- **Better environment isolation**: No more accidental production deployments
- **Clearer configuration**: Environment-specific configs are easier to manage
- **Improved security**: Sensitive data in environment-specific files
- **Faster development**: Less confusion about which environment is active
- **Modern architecture**: Aligned with industry best practices

## ⏰ Action Required

**Important**: Legacy support will be completely removed in v2.0. Please migrate your projects before the deadline to ensure uninterrupted service.

### Recommended Timeline

- **Now - v1.10**: Migrate development environments
- **v1.10 - v2.0**: Migrate production environments
- **v2.0**: Legacy code removed

---

*Last Updated: 2025-08-31*  
*Version: 1.9*  
*Status: Active Deprecation Phase*