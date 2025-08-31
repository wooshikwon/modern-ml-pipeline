#!/bin/bash

# Modern ML Pipeline v2.0 - Legacy Code Cleanup Script
# This script removes all legacy files and directories for v2.0 release

echo "🧹 Modern ML Pipeline v2.0 - Cleaning up legacy files..."
echo "================================================"

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 1. Remove legacy directories
echo ""
echo "1️⃣ Removing legacy directories..."

if [ -d "config" ]; then
    echo "   ❌ Removing legacy config/ directory..."
    rm -rf config
    echo "   ✅ Removed config/"
else
    echo "   ✓ No config/ directory found"
fi

if [ -d "models/recipes" ]; then
    echo "   ❌ Removing legacy models/recipes/ directory..."
    rm -rf models/recipes
    echo "   ✅ Removed models/recipes/"
else
    echo "   ✓ No models/recipes/ directory found"
fi

# 2. Remove legacy files
echo ""
echo "2️⃣ Removing legacy files..."

legacy_files=(
    ".env"
    "settings_old.py"
    "src/settings/_legacy.py"
    "src/cli/compat.py"
    "tests/test_legacy.py"
    "src/utils/deprecation.py"
    "src/cli/commands/migrate_command.py"
    "DEPRECATION.md"
    ".mmp_legacy_check"
)

removed_count=0
for file in "${legacy_files[@]}"; do
    if [ -f "$file" ]; then
        echo "   ❌ Removing $file..."
        rm "$file"
        ((removed_count++))
    fi
done

if [ $removed_count -eq 0 ]; then
    echo "   ✓ No legacy files found"
else
    echo "   ✅ Removed $removed_count legacy files"
fi

# 3. Clean Python cache
echo ""
echo "3️⃣ Cleaning Python cache..."

# Count cache files before removal
cache_count=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
pyc_count=$(find . -type f -name "*.pyc" 2>/dev/null | wc -l)

# Remove cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find . -type f -name ".coverage*" -delete 2>/dev/null

echo "   ✅ Removed $cache_count __pycache__ directories"
echo "   ✅ Removed $pyc_count .pyc files"

# 4. Remove any remaining deprecation markers
echo ""
echo "4️⃣ Scanning for deprecation markers in code..."

deprecation_count=$(grep -r "deprecated\|DEPRECATED\|@deprecated" src/ tests/ --include="*.py" 2>/dev/null | wc -l)

if [ $deprecation_count -gt 0 ]; then
    echo "   ⚠️  Found $deprecation_count deprecation markers still in code"
    echo "   Please review these manually:"
    grep -r "deprecated\|DEPRECATED\|@deprecated" src/ tests/ --include="*.py" 2>/dev/null | head -5
else
    echo "   ✓ No deprecation markers found"
fi

# 5. Final summary
echo ""
echo "================================================"
echo "✅ Cleanup complete!"
echo ""
echo "Next steps:"
echo "  1. Run tests: uv run pytest tests/"
echo "  2. Check for import errors: uv run python -c 'from src.settings import Settings'"
echo "  3. Commit changes: git add -A && git commit -m 'chore: remove legacy code for v2.0'"
echo ""
echo "🎉 Your codebase is now ready for v2.0 release!"