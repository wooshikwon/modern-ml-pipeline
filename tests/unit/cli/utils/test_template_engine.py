"""
Template Engine Unit Tests - No Mock Hell Approach
Real template rendering, real file operations
Following comprehensive testing strategy document principles
"""

import tempfile
from pathlib import Path

import pytest
from jinja2 import TemplateNotFound

from src.cli.utils.template_engine import TemplateEngine


class TestTemplateEngine:
    """Test TemplateEngine with real template files and rendering."""

    @pytest.fixture
    def template_dir(self, tmp_path):
        """Create a temporary template directory with sample templates."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()

        # Create sample templates
        simple_template = template_dir / "simple.j2"
        simple_template.write_text("Hello {{ name }}!")

        config_template = template_dir / "config.yaml.j2"
        config_template.write_text(
            """
name: {{ project_name }}
version: {{ version }}
features:
{% for feature in features %}
  - {{ feature }}
{% endfor %}
""".strip()
        )

        nested_dir = template_dir / "nested"
        nested_dir.mkdir()
        nested_template = nested_dir / "nested.txt.j2"
        nested_template.write_text("Nested: {{ value }}")

        # Create a static file (non-template)
        static_file = template_dir / "static.txt"
        static_file.write_text("Static content")

        return template_dir

    def test_initialization_with_valid_directory(self, template_dir):
        """Test TemplateEngine initialization with valid directory."""
        # Given: A valid template directory
        # When: Creating TemplateEngine
        engine = TemplateEngine(template_dir)

        # Then: Engine is properly initialized
        assert engine.template_dir == template_dir
        assert engine.env is not None
        assert engine.env.loader is not None

    def test_initialization_with_invalid_directory(self, tmp_path):
        """Test TemplateEngine initialization with non-existent directory."""
        # Given: A non-existent directory
        invalid_dir = tmp_path / "non_existent"

        # When/Then: Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Template directory not found"):
            TemplateEngine(invalid_dir)

    def test_render_simple_template(self, template_dir):
        """Test rendering a simple template."""
        # Given: TemplateEngine and context
        engine = TemplateEngine(template_dir)
        context = {"name": "World"}

        # When: Rendering template
        result = engine.render_template("simple.j2", context)

        # Then: Template is rendered correctly
        assert result == "Hello World!"

    def test_render_complex_template(self, template_dir):
        """Test rendering a complex template with loops."""
        # Given: TemplateEngine and complex context
        engine = TemplateEngine(template_dir)
        context = {
            "project_name": "ML Pipeline",
            "version": "1.0.0",
            "features": ["training", "evaluation", "serving"],
        }

        # When: Rendering template
        result = engine.render_template("config.yaml.j2", context)

        # Then: Template is rendered correctly
        assert "name: ML Pipeline" in result
        assert "version: 1.0.0" in result
        assert "- training" in result
        assert "- evaluation" in result
        assert "- serving" in result

    def test_render_nested_template(self, template_dir):
        """Test rendering template in nested directory."""
        # Given: TemplateEngine and context
        engine = TemplateEngine(template_dir)
        context = {"value": "test_value"}

        # When: Rendering nested template
        result = engine.render_template("nested/nested.txt.j2", context)

        # Then: Template is rendered correctly
        assert result == "Nested: test_value"

    def test_render_non_existent_template(self, template_dir):
        """Test rendering non-existent template raises error."""
        # Given: TemplateEngine
        engine = TemplateEngine(template_dir)

        # When/Then: Should raise TemplateNotFound
        with pytest.raises(TemplateNotFound):
            engine.render_template("non_existent.j2", {})

    def test_write_rendered_file(self, template_dir, tmp_path):
        """Test writing rendered template to file."""
        # Given: TemplateEngine and output path
        engine = TemplateEngine(template_dir)
        output_path = tmp_path / "output" / "result.txt"
        context = {"name": "Test"}

        # When: Writing rendered file
        engine.write_rendered_file("simple.j2", output_path, context)

        # Then: File is created with correct content
        assert output_path.exists()
        assert output_path.read_text() == "Hello Test!"

    def test_write_rendered_file_without_create_dirs(self, template_dir, tmp_path):
        """Test writing file without creating directories."""
        # Given: TemplateEngine and non-existent parent directory
        engine = TemplateEngine(template_dir)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_path = output_dir / "result.txt"
        context = {"name": "Test"}

        # When: Writing without creating dirs
        engine.write_rendered_file("simple.j2", output_path, context, create_dirs=False)

        # Then: File is created
        assert output_path.exists()

    def test_write_rendered_file_io_error(self, template_dir):
        """Test write failure raises IOError."""
        # Given: TemplateEngine and read-only directory
        engine = TemplateEngine(template_dir)

        # Create a read-only directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.txt"
            output_path.touch()
            output_path.chmod(0o444)  # Read-only

            # When/Then: Should raise IOError
            with pytest.raises(IOError):
                engine.write_rendered_file("simple.j2", output_path, {"name": "Test"})

    def test_copy_static_file(self, template_dir, tmp_path):
        """Test copying static file."""
        # Given: TemplateEngine and output path
        engine = TemplateEngine(template_dir)
        output_path = tmp_path / "copied" / "static.txt"

        # When: Copying static file
        engine.copy_static_file("static.txt", output_path)

        # Then: File is copied
        assert output_path.exists()
        assert output_path.read_text() == "Static content"

    def test_copy_static_file_non_existent(self, template_dir, tmp_path):
        """Test copying non-existent file raises error."""
        # Given: TemplateEngine
        engine = TemplateEngine(template_dir)
        output_path = tmp_path / "output.txt"

        # When/Then: Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Source file not found"):
            engine.copy_static_file("non_existent.txt", output_path)

    def test_copy_static_file_without_create_dirs(self, template_dir, tmp_path):
        """Test copying file without creating directories."""
        # Given: TemplateEngine and existing output directory
        engine = TemplateEngine(template_dir)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        output_path = output_dir / "static.txt"

        # When: Copying without creating dirs
        engine.copy_static_file("static.txt", output_path, create_dirs=False)

        # Then: File is copied
        assert output_path.exists()

    def test_list_templates_all(self, template_dir):
        """Test listing all templates."""
        # Given: TemplateEngine with templates
        engine = TemplateEngine(template_dir)

        # When: Listing all templates
        templates = engine.list_templates()

        # Then: All files are listed
        assert "simple.j2" in templates
        assert "config.yaml.j2" in templates
        assert "nested/nested.txt.j2" in templates
        assert "static.txt" in templates
        assert len(templates) == 4

    def test_list_templates_with_pattern(self, template_dir):
        """Test listing templates with pattern."""
        # Given: TemplateEngine
        engine = TemplateEngine(template_dir)

        # When: Listing templates with .j2 pattern
        templates = engine.list_templates("*.j2")

        # Then: Only .j2 files are listed
        assert "simple.j2" in templates
        assert "config.yaml.j2" in templates
        assert "static.txt" not in templates

    def test_list_templates_nested_pattern(self, template_dir):
        """Test listing templates with nested pattern."""
        # Given: TemplateEngine
        engine = TemplateEngine(template_dir)

        # When: Listing templates in nested directory
        templates = engine.list_templates("nested/*.j2")

        # Then: Only nested templates are listed
        assert templates == ["nested/nested.txt.j2"]

    def test_template_with_empty_context(self, template_dir):
        """Test rendering template with empty context."""
        # Given: Template without variables
        no_vars_template = template_dir / "no_vars.j2"
        no_vars_template.write_text("No variables here")

        engine = TemplateEngine(template_dir)

        # When: Rendering with empty context
        result = engine.render_template("no_vars.j2", {})

        # Then: Template renders successfully
        assert result == "No variables here"

    def test_template_with_missing_variable(self, template_dir):
        """Test that missing variables in context work gracefully (Jinja2 default behavior)."""
        # Given: TemplateEngine with incomplete context
        engine = TemplateEngine(template_dir)
        context = {}  # Missing 'name' variable

        # When: Rendering template with missing variable
        result = engine.render_template("simple.j2", context)

        # Then: Jinja2 renders missing variables as empty string
        assert result == "Hello !"

    def test_jinja2_environment_configuration(self, template_dir):
        """Test Jinja2 environment is configured correctly."""
        # Given: TemplateEngine
        engine = TemplateEngine(template_dir)

        # When: Checking environment configuration
        env = engine.env

        # Then: Environment has expected configuration
        assert env.autoescape == False
        assert env.trim_blocks == True
        assert env.lstrip_blocks == True
        assert env.keep_trailing_newline == True

    def test_template_with_whitespace_control(self, template_dir):
        """Test template rendering with whitespace control."""
        # Given: Template with whitespace control markers
        ws_template = template_dir / "whitespace.j2"
        ws_template.write_text(
            "{% for item in items %}{{ item }}{% if not loop.last %} {% endif %}{% endfor %}"
        )

        engine = TemplateEngine(template_dir)
        context = {"items": ["a", "b", "c"]}

        # When: Rendering template
        result = engine.render_template("whitespace.j2", context)

        # Then: Whitespace is controlled properly
        assert result == "a b c"

    def test_copy_static_file_io_error(self, template_dir):
        """Test copy failure raises IOError."""
        # Given: TemplateEngine and problematic destination
        engine = TemplateEngine(template_dir)

        # Try to copy to a file that can't be written
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "readonly"
            output_dir.mkdir()
            output_dir.chmod(0o444)  # Read-only directory
            output_path = output_dir / "static.txt"

            # When/Then: Should raise IOError
            with pytest.raises(IOError):
                engine.copy_static_file("static.txt", output_path)
