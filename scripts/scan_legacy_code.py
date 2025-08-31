#!/usr/bin/env python3
"""
Modern ML Pipeline v2.0 - Legacy Code Scanner
스캔을 통해 남아있는 레거시 코드 패턴을 찾습니다.
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()


class LegacyCodeScanner(ast.NodeVisitor):
    """레거시 코드 패턴을 찾는 AST Visitor"""
    
    def __init__(self):
        self.legacy_patterns: List[Tuple[int, str]] = []
        self.deprecated_imports = [
            "load_config_files",  # Without env_name
            "MLTaskSettings",  # Old name
            "app_env",  # Removed field
            "get_env_name_with_fallback",  # Removed function
            "show_deprecation_warning",  # Removed function
            "migrate_command",  # Removed command
        ]
        self.legacy_directories = ["config", "models/recipes"]
        self.legacy_env_vars = ["APP_ENV"]
    
    def visit_Name(self, node: ast.Name) -> None:
        """변수명 체크"""
        if node.id in ["app_env", "gcp_project_id"]:
            self.legacy_patterns.append(
                (node.lineno, f"Legacy variable: {node.id}")
            )
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute) -> None:
        """속성 접근 체크"""
        if node.attr in ["app_env"]:
            self.legacy_patterns.append(
                (node.lineno, f"Legacy attribute: {node.attr}")
            )
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """import 문 체크"""
        if node.module:
            if "legacy" in node.module or "deprecated" in node.module:
                self.legacy_patterns.append(
                    (node.lineno, f"Legacy import: {node.module}")
                )
            
            # Check for specific deprecated imports
            for alias in node.names:
                if alias.name in self.deprecated_imports:
                    self.legacy_patterns.append(
                        (node.lineno, f"Deprecated import: {alias.name}")
                    )
        self.generic_visit(node)
    
    def visit_Constant(self, node: ast.Constant) -> None:
        """문자열 상수 체크"""
        if isinstance(node.value, str):
            # Check for legacy paths
            if "config/" in node.value and "configs/" not in node.value:
                self.legacy_patterns.append(
                    (node.lineno, f"Legacy path reference: config/")
                )
            if "models/recipes/" in node.value:
                self.legacy_patterns.append(
                    (node.lineno, f"Legacy path reference: models/recipes/")
                )
            # Check for legacy env vars
            if node.value in self.legacy_env_vars:
                self.legacy_patterns.append(
                    (node.lineno, f"Legacy env var: {node.value}")
                )
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """함수 호출 체크"""
        # Check for os.getenv('APP_ENV')
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr == 'getenv' and 
            len(node.args) > 0 and
            isinstance(node.args[0], ast.Constant) and
            node.args[0].value == 'APP_ENV'):
            self.legacy_patterns.append(
                (node.lineno, "Legacy env var access: APP_ENV")
            )
        
        # Check for deprecated function calls
        if isinstance(node.func, ast.Name):
            if node.func.id in self.deprecated_imports:
                self.legacy_patterns.append(
                    (node.lineno, f"Deprecated function call: {node.func.id}")
                )
        
        self.generic_visit(node)


def scan_file(file_path: Path) -> List[Tuple[int, str]]:
    """단일 Python 파일 스캔"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content)
            scanner = LegacyCodeScanner()
            scanner.visit(tree)
            return scanner.legacy_patterns
    except (SyntaxError, UnicodeDecodeError):
        return []


def scan_project(root_dir: Path = None) -> Dict[Path, List[Tuple[int, str]]]:
    """프로젝트 전체 스캔"""
    if root_dir is None:
        root_dir = Path.cwd()
    
    results = {}
    
    # Find all Python files
    py_files = list(root_dir.glob("src/**/*.py")) + list(root_dir.glob("tests/**/*.py"))
    
    console.print(f"🔍 Scanning {len(py_files)} Python files for legacy patterns...")
    console.print()
    
    for py_file in track(py_files, description="Scanning..."):
        patterns = scan_file(py_file)
        if patterns:
            results[py_file] = patterns
    
    return results


def check_directory_structure(root_dir: Path = None) -> List[str]:
    """레거시 디렉토리 구조 체크"""
    if root_dir is None:
        root_dir = Path.cwd()
    
    issues = []
    
    # Check for legacy directories
    if (root_dir / "config").exists():
        issues.append("Legacy directory found: config/")
    
    if (root_dir / "models" / "recipes").exists():
        issues.append("Legacy directory found: models/recipes/")
    
    # Check for .env file (should be .env.{env_name})
    if (root_dir / ".env").exists():
        issues.append("Legacy .env file found (should be .env.{env_name})")
    
    return issues


def display_results(results: Dict[Path, List[Tuple[int, str]]], 
                   dir_issues: List[str]) -> None:
    """결과를 보기 좋게 출력"""
    
    # Directory structure issues
    if dir_issues:
        console.print("⚠️  [bold yellow]Legacy Directory Structure Issues:[/bold yellow]")
        for issue in dir_issues:
            console.print(f"   • {issue}")
        console.print()
    
    # Code pattern issues
    if results:
        console.print("⚠️  [bold yellow]Legacy Code Patterns Found:[/bold yellow]")
        console.print()
        
        # Create summary table
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("File", style="cyan")
        table.add_column("Line", justify="right", style="green")
        table.add_column("Issue", style="red")
        
        total_issues = 0
        for file_path, patterns in results.items():
            rel_path = file_path.relative_to(Path.cwd())
            for line_no, pattern in patterns[:3]:  # Show max 3 per file
                table.add_row(str(rel_path), str(line_no), pattern)
                total_issues += 1
            
            if len(patterns) > 3:
                table.add_row(str(rel_path), "...", f"and {len(patterns) - 3} more")
                total_issues += len(patterns) - 3
        
        console.print(table)
        console.print()
        console.print(f"📊 Total issues found: [bold red]{total_issues}[/bold red]")
        console.print()
        console.print("💡 [yellow]To fix these issues:[/yellow]")
        console.print("   1. Run: [bold]bash scripts/cleanup_legacy.sh[/bold]")
        console.print("   2. Manually review and update the remaining code patterns")
        console.print("   3. Update imports to use v2.0 APIs")
    else:
        if not dir_issues:
            console.print("✅ [bold green]No legacy code patterns found![/bold green]")
            console.print("🎉 Your codebase is clean and ready for v2.0")


def main():
    """메인 실행 함수"""
    console.print("[bold blue]Modern ML Pipeline v2.0 - Legacy Code Scanner[/bold blue]")
    console.print("=" * 50)
    console.print()
    
    # Check directory structure
    dir_issues = check_directory_structure()
    
    # Scan code
    results = scan_project()
    
    # Display results
    display_results(results, dir_issues)
    
    # Exit code
    if results or dir_issues:
        sys.exit(1)  # Found issues
    else:
        sys.exit(0)  # Clean


if __name__ == "__main__":
    main()