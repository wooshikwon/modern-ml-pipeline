"""
Deprecation Tests
Phase 4: Test deprecation warnings and migration tools
"""

import warnings
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from src.utils.deprecation import deprecated, show_deprecation_warning, DeprecatedProperty
from src.settings._config_schema import EnvironmentSettings
from src.cli.commands.migrate_command import (
    MigrationTask, 
    check_legacy_structure, 
    analyze_project_structure
)


class TestDeprecationUtility:
    """Test deprecation utility functions."""
    
    def test_deprecated_function(self):
        """Test deprecated decorator on function."""
        @deprecated(
            reason="This function is outdated",
            version="2.0",
            alternative="use_new_function()"
        )
        def old_function():
            return "old"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = old_function()
            
            assert result == "old"
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()
            assert "use_new_function()" in str(w[0].message)
            assert "2.0" in str(w[0].message)
    
    def test_show_deprecation_warning(self):
        """Test show_deprecation_warning function."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            show_deprecation_warning(
                "Old feature",
                version="3.0",
                alternative="New feature"
            )
            
            assert len(w) == 1
            assert "Old feature" in str(w[0].message)
            assert "3.0" in str(w[0].message)
            assert "New feature" in str(w[0].message)
    
    def test_critical_deprecation_warning(self):
        """Test critical deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            show_deprecation_warning(
                "Critical feature",
                critical=True
            )
            
            assert len(w) == 1
            assert "CRITICAL" in str(w[0].message)


class TestSettingsDeprecation:
    """Test deprecation in Settings module."""
    
    def test_app_env_deprecation(self):
        """Test app_env field shows deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Create settings with app_env
            settings = EnvironmentSettings(
                app_env="dev",
                gcp_project_id="test-project"
            )
            
            # Should warn when app_env is set
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            assert any("app_env" in msg for msg in warning_messages)
            assert any("deprecated" in msg.lower() for msg in warning_messages)
    
    def test_hardcoded_gcp_project_warning(self):
        """Test warning for hardcoded GCP project ID."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            settings = EnvironmentSettings(
                gcp_project_id="hardcoded-project"
            )
            
            # Should warn about hardcoded project ID
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            assert any("Hardcoding" in msg for msg in warning_messages)
    
    def test_no_warning_for_env_var_gcp_project(self):
        """Test no warning when using environment variable."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            settings = EnvironmentSettings(
                gcp_project_id="${GCP_PROJECT}"
            )
            
            # Should not warn for environment variable
            hardcoding_warnings = [
                warning for warning in w 
                if "Hardcoding" in str(warning.message)
            ]
            assert len(hardcoding_warnings) == 0


class TestCLIDeprecation:
    """Test CLI command deprecation warnings."""
    
    @pytest.fixture
    def cli_runner(self):
        """Create CLI test runner."""
        return CliRunner()
    
    def test_train_without_env_name(self, cli_runner):
        """Test train command without env_name shows warning."""
        from src.cli.main_commands import app
        
        with patch('src.cli.main_commands.load_environment'):
            with patch('src.cli.main_commands.load_settings_by_file'):
                with patch('src.cli.main_commands.run_training'):
                    with patch('src.cli.utils.env_loader.get_env_name_with_fallback', return_value='test'):
                        result = cli_runner.invoke(
                            app, 
                            ["train", "--recipe-file", "test.yaml"]
                        )
                        
                        # Should show deprecation warning
                        assert "WARNING" in result.output
                        assert "deprecated" in result.output.lower()
                        assert "--env-name" in result.output
    
    def test_batch_inference_without_env_name(self, cli_runner):
        """Test batch-inference command without env_name shows warning."""
        from src.cli.main_commands import app
        
        with patch('src.cli.main_commands.load_environment'):
            with patch('src.cli.main_commands.load_config_files'):
                with patch('src.cli.main_commands.create_settings_for_inference'):
                    with patch('src.cli.main_commands.run_batch_inference'):
                        with patch('src.cli.utils.env_loader.get_env_name_with_fallback', return_value='test'):
                            result = cli_runner.invoke(
                                app,
                                ["batch-inference", "--run-id", "test123"]
                            )
                            
                            # Should show deprecation warning
                            assert "WARNING" in result.output
                            assert "deprecated" in result.output.lower()
                            assert "--env-name" in result.output


class TestMigrationCommand:
    """Test migration assistant command."""
    
    def test_check_legacy_structure(self):
        """Test legacy structure detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            # Create legacy structure
            Path("config").mkdir()
            Path(".env").touch()
            Path("models/recipes").mkdir(parents=True)
            
            legacy_items = check_legacy_structure()
            
            assert len(legacy_items) >= 2
            assert any("config" in item[0] for item in legacy_items)
            assert any(".env" in item[0] for item in legacy_items)
    
    def test_analyze_project_structure(self):
        """Test project structure analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            # Create legacy structure
            Path("config").mkdir()
            Path("models/recipes").mkdir(parents=True)
            Path(".env").write_text("ENV_NAME=dev\n")
            
            tasks = analyze_project_structure()
            
            assert len(tasks) > 0
            assert any(task.action == "rename" for task in tasks)
            assert any("config" in task.description for task in tasks)
    
    def test_migration_task_execution(self):
        """Test migration task execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = Path(tmpdir) / "old_file.txt"
            new_path = Path(tmpdir) / "new_file.txt"
            old_path.write_text("content")
            
            task = MigrationTask(
                description="Rename file",
                old_path=old_path,
                new_path=new_path,
                action="rename"
            )
            
            # Test dry run
            assert task.execute(dry_run=True)
            assert old_path.exists()  # Should not be renamed
            
            # Test actual execution
            assert task.execute(dry_run=False)
            assert not old_path.exists()
            assert new_path.exists()
            assert new_path.read_text() == "content"
    
    def test_migrate_command_dry_run(self):
        """Test migrate command in dry-run mode."""
        from src.cli.main_commands import app
        
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            
            # Create legacy structure
            Path("config").mkdir()
            
            result = runner.invoke(
                app,
                ["migrate", "--dry-run", "--force"]
            )
            
            if result.exit_code != 0:
                print(f"Exit code: {result.exit_code}")
                print(f"Output: {result.output}")
                print(f"Exception: {result.exception}")
            
            assert result.exit_code == 0
            assert "Migration Plan" in result.output
            assert "Dry run mode" in result.output
            
            # Config should still exist (dry run)
            assert Path("config").exists()


class TestLoadersDeprecation:
    """Test deprecation in loaders module."""
    
    def test_load_settings_without_env_name(self):
        """Test load_settings_by_file without env_name shows warning."""
        from src.settings.loaders import load_settings_by_file
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            with patch('src.settings.loaders.load_config_files'):
                with patch('src.settings.loaders.load_recipe_file'):
                    with patch('src.settings.loaders._is_modern_recipe_structure', return_value=False):
                        try:
                            load_settings_by_file("test.yaml")
                        except:
                            pass  # We're just testing warnings
            
            # Should show deprecation warning
            assert len(w) >= 1
            warning_messages = [str(warning.message) for warning in w]
            assert any("without env_name" in msg for msg in warning_messages)