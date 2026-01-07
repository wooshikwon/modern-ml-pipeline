"""
Template Engine for Modern ML Pipeline CLI
Phase 1: Clean Jinja2-based template rendering system

CLAUDE.md 원칙 준수:
- 타입 힌트 필수
- Google Style Docstring
- 단일 책임 원칙
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

logger = logging.getLogger(__name__)


class TemplateEngine:
    """Jinja2 기반 템플릿 렌더링 엔진.

    프로젝트 초기화 및 설정 파일 생성을 위한 템플릿 렌더링 기능 제공.
    """

    def __init__(self, template_dir: Path):
        """템플릿 엔진 초기화.

        Args:
            template_dir: 템플릿 파일이 위치한 디렉토리 경로

        Raises:
            FileNotFoundError: 템플릿 디렉토리가 존재하지 않을 경우
        """
        if not template_dir.exists():
            raise FileNotFoundError(f"Template directory not found: {template_dir}")

        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """템플릿 파일을 렌더링하여 문자열로 반환.

        Args:
            template_name: 렌더링할 템플릿 파일 이름 (상대 경로)
            context: 템플릿에 전달할 변수 딕셔너리

        Returns:
            렌더링된 템플릿 문자열

        Raises:
            TemplateNotFound: 템플릿 파일을 찾을 수 없을 경우
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except TemplateNotFound:
            logger.error(f"Template을 찾을 수 없습니다: {template_name}")
            raise

    def write_rendered_file(
        self,
        template_name: str,
        output_path: Path,
        context: Dict[str, Any],
        create_dirs: bool = True,
    ) -> None:
        """렌더링된 템플릿을 파일로 저장.

        Args:
            template_name: 렌더링할 템플릿 파일 이름
            output_path: 출력 파일 경로
            context: 템플릿에 전달할 변수 딕셔너리
            create_dirs: 상위 디렉토리 자동 생성 여부

        Raises:
            TemplateNotFound: 템플릿 파일을 찾을 수 없을 경우
            IOError: 파일 쓰기 실패 시
        """
        # 상위 디렉토리 생성
        if create_dirs:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # 템플릿 렌더링 및 파일 쓰기
        rendered_content = self.render_template(template_name, context)

        try:
            output_path.write_text(rendered_content, encoding="utf-8")
        except IOError as e:
            logger.error(f"파일 작성에 실패했습니다: {output_path}, 오류: {e}")
            raise

    def copy_static_file(
        self, source_name: str, output_path: Path, create_dirs: bool = True
    ) -> None:
        """정적 파일 복사 (템플릿 렌더링 없이).

        Args:
            source_name: 복사할 소스 파일 이름 (템플릿 디렉토리 내 상대 경로)
            output_path: 출력 파일 경로
            create_dirs: 상위 디렉토리 자동 생성 여부

        Raises:
            FileNotFoundError: 소스 파일을 찾을 수 없을 경우
            IOError: 파일 복사 실패 시
        """
        source_path = self.template_dir / source_name

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # 상위 디렉토리 생성
        if create_dirs:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import shutil

            shutil.copy2(source_path, output_path)
            logger.debug(f"파일이 복사되었습니다: {output_path}")
        except IOError as e:
            logger.error(f"파일 복사에 실패했습니다: {source_path} -> {output_path}, 오류: {e}")
            raise

    def list_templates(self, pattern: Optional[str] = None) -> list[str]:
        """사용 가능한 템플릿 파일 목록 반환.

        Args:
            pattern: 파일 패턴 (예: "*.j2", "configs/*.yaml.j2")

        Returns:
            템플릿 파일 이름 리스트
        """
        templates = []

        if pattern:
            template_paths = self.template_dir.glob(pattern)
        else:
            template_paths = self.template_dir.rglob("*")

        for path in template_paths:
            if path.is_file():
                relative_path = path.relative_to(self.template_dir)
                templates.append(str(relative_path))

        return sorted(templates)
