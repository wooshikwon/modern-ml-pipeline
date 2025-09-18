#!/usr/bin/env python3
"""
포괄적 테스트 실행 및 메트릭 분석 도구

16개 그룹으로 세분화된 테스트를 순차/병렬 실행하며
커버리지, 에러율, 스킵율, 실행속도를 점증적으로 수집합니다.
"""

import subprocess
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import sys
import argparse


@dataclass
class TestResult:
    """단일 테스트 그룹 실행 결과"""
    group_id: int
    group_name: str
    phase: int
    paths: List[str]
    command: str
    start_time: str
    end_time: str
    duration_seconds: float
    total_tests: int
    passed: int
    failed: int
    skipped: int
    error_rate: float
    skip_rate: float
    coverage_delta: Optional[float] = None
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class PhaseResult:
    """Phase 실행 결과"""
    phase_id: int
    phase_name: str
    groups: List[TestResult]
    total_duration: float
    total_tests: int
    total_passed: int
    total_failed: int
    total_skipped: int
    phase_error_rate: float
    phase_skip_rate: float


class ComprehensiveTestRunner:
    """포괄적 테스트 실행기"""
    
    def __init__(self, project_root: Path, output_file: Optional[Path] = None):
        self.project_root = project_root
        self.output_file = output_file or project_root / "test_metrics_comprehensive.json"
        self.logger = self._setup_logging()
        self.results: List[TestResult] = []
        self.phase_results: List[PhaseResult] = []
        self.cumulative_coverage = 0.0
        
        # 테스트 그룹 정의
        self.test_groups = self._define_test_groups()
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger("comprehensive_test_runner")
        logger.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 파일 핸들러
        log_file = self.project_root / "test_runner.log"
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 포매터
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
        
    def _define_test_groups(self) -> List[Dict[str, Any]]:
        """16개 테스트 그룹 정의"""
        return [
            # Phase 1: 빠른 단위 테스트
            {
                "id": 1, "name": "CLI 핵심 기능", "phase": 1,
                "paths": ["tests/unit/cli/"],
                "parallel": True, "timeout": 120
            },
            {
                "id": 2, "name": "CLI 유틸리티", "phase": 1,
                "paths": ["tests/unit/cli/utils/"],
                "parallel": True, "timeout": 90
            },
            {
                "id": 3, "name": "설정 및 팩토리", "phase": 1,
                "paths": ["tests/unit/settings/", "tests/unit/factory/"],
                "parallel": True, "timeout": 120
            },
            {
                "id": 4, "name": "기본 어댑터 및 페처", "phase": 1,
                "paths": ["tests/unit/components/adapters/", "tests/unit/components/fetchers/"],
                "parallel": True, "timeout": 90
            },
            {
                "id": 5, "name": "유틸리티 핵심", "phase": 1,
                "paths": ["tests/unit/utils/core/", "tests/unit/utils/data/"],
                "parallel": True, "timeout": 90
            },
            
            # Phase 2: 중간 단위 테스트
            {
                "id": 6, "name": "전처리 컴포넌트", "phase": 2,
                "paths": ["tests/unit/components/preprocessor/"],
                "parallel": False, "timeout": 180, "workers": 2
            },
            {
                "id": 7, "name": "데이터 핸들러", "phase": 2,
                "paths": ["tests/unit/components/datahandlers/"],
                "parallel": False, "timeout": 120, "workers": 2
            },
            {
                "id": 8, "name": "모델 컴포넌트", "phase": 2,
                "paths": ["tests/unit/components/models/", "tests/unit/models/custom/"],
                "parallel": False, "timeout": 300, "workers": 1
            },
            {
                "id": 9, "name": "평가자 및 캘리브레이션", "phase": 2,
                "paths": ["tests/unit/components/evaluators/", "tests/unit/components/calibration/"],
                "parallel": False, "timeout": 150, "workers": 2
            },
            {
                "id": 10, "name": "트레이너 및 최적화", "phase": 2,
                "paths": ["tests/unit/components/trainer/"],
                "parallel": False, "timeout": 120
            },
            {
                "id": 11, "name": "파이프라인", "phase": 2,
                "paths": ["tests/unit/pipelines/"],
                "parallel": False, "timeout": 300, "workers": 1
            },
            {
                "id": 12, "name": "서빙", "phase": 2,
                "paths": ["tests/unit/serving/"],
                "parallel": False, "timeout": 150, "workers": 2
            },
            {
                "id": 13, "name": "유틸리티 통합", "phase": 2,
                "paths": [
                    "tests/unit/utils/integrations/", "tests/unit/utils/mlflow/",
                    "tests/unit/utils/system/", "tests/unit/utils/template/",
                    "tests/unit/utils/deps/", "tests/unit/utils/database/"
                ],
                "parallel": False, "timeout": 240, "workers": 1
            },
            
            # Phase 3: 통합 및 E2E 테스트
            {
                "id": 14, "name": "데이터베이스 및 MLFlow 통합", "phase": 3,
                "paths": [
                    "tests/integration/test_database_integration.py",
                    "tests/integration/test_mlflow_integration.py",
                    "tests/integration/test_settings_integration.py"
                ],
                "parallel": False, "timeout": 600, "isolated": True
            },
            {
                "id": 15, "name": "컴포넌트 상호작용 및 오류 전파", "phase": 3,
                "paths": [
                    "tests/integration/test_component_interactions.py",
                    "tests/integration/test_error_propagation.py",
                    "tests/integration/test_integration_completeness.py",
                    "tests/integration/test_production_readiness.py"
                ],
                "parallel": False, "timeout": 800, "isolated": True
            },
            {
                "id": 16, "name": "파이프라인 통합 및 서빙", "phase": 3,
                "paths": [
                    "tests/integration/test_cli_pipeline_integration.py",
                    "tests/integration/test_inference_pipeline_integration.py",
                    "tests/integration/test_pipeline_orchestration.py",
                    "tests/integration/test_preprocessor_pipeline_integration.py",
                    "tests/integration/test_serving*.py",
                    "tests/e2e/"
                ],
                "parallel": False, "timeout": 1200, "isolated": True
            }
        ]
    
    def _build_pytest_command(self, group: Dict[str, Any], verbose: bool = True) -> str:
        """그룹에 맞는 pytest 명령어 생성"""
        paths = " ".join(group["paths"])

        base_cmd = f"pytest {paths}"
        options = []

        # 기본 옵션
        if verbose:
            options.append("-v --tb=short")
        else:
            options.append("--tb=line")

        # 병렬 처리 (커버리지와 병렬 실행 시 문제가 있으므로 조정)
        if group.get("parallel", False):
            # 커버리지를 사용할 때는 병렬 처리 제한
            options.append("-n 2")  # auto 대신 2로 제한
        elif group.get("workers"):
            options.append(f"-n {group['workers']}")
        else:
            options.append("-n 1")

        # 타임아웃
        if group.get("timeout"):
            options.append(f"--timeout={group['timeout']}")

        # 에러 처리
        if group["phase"] == 3:  # Phase 3은 격리 실행
            options.append("-x --maxfail=2")
        elif group["phase"] == 2:  # Phase 2는 중간 관용
            options.append("--maxfail=10")
        else:  # Phase 1은 관대
            options.append("--maxfail=15")

        # 커버리지 (병렬 실행 시 데이터 파일 분리)
        options.append("--cov=src")
        options.append(f"--cov-report=html:htmlcov/group{group['id']}")
        options.append("--cov-report=term-missing")

        return f"{base_cmd} {' '.join(options)}"
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, int]:
        """pytest 출력에서 테스트 결과 파싱"""
        results = {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "warnings": 0}

        # stdout과 stderr 모두에서 결과 찾기
        combined_output = stdout + "\n" + stderr
        lines = combined_output.split('\n')

        for line in lines:
            line = line.strip()

            # Pattern 1: "X failed, Y passed, Z skipped, W warnings in TIME"
            # Pattern 2: "X passed, Y skipped, Z warnings in TIME"
            # Pattern 3: "X passed in TIME"
            if " in " in line and ("passed" in line or "failed" in line):
                # 숫자와 키워드 추출
                import re

                # failed 찾기
                failed_match = re.search(r'(\d+)\s+failed', line)
                if failed_match:
                    results["failed"] = int(failed_match.group(1))

                # passed 찾기
                passed_match = re.search(r'(\d+)\s+passed', line)
                if passed_match:
                    results["passed"] = int(passed_match.group(1))

                # skipped 찾기
                skipped_match = re.search(r'(\d+)\s+skipped', line)
                if skipped_match:
                    results["skipped"] = int(skipped_match.group(1))

                # warnings 찾기 (참고용)
                warnings_match = re.search(r'(\d+)\s+warning', line)
                if warnings_match:
                    results["warnings"] = int(warnings_match.group(1))

                # 총계 계산
                results["total"] = results["passed"] + results["failed"] + results["skipped"]

                # 결과를 찾았으면 종료
                if results["total"] > 0:
                    break

        # 아무 결과도 못 찾은 경우, exit code로 판단
        if results["total"] == 0:
            # INTERNALERROR나 에러가 있었는지 확인
            if "INTERNALERROR" in combined_output:
                # Coverage 에러 등으로 인한 실패
                # 하지만 실제 테스트는 실행되었을 수 있으므로 다른 패턴 찾기
                for line in lines:
                    if "==" in line and ("passed" in line or "failed" in line):
                        # 좀 더 유연한 패턴 매칭
                        import re
                        numbers = re.findall(r'\d+', line)
                        keywords = re.findall(r'(passed|failed|skipped)', line)

                        if numbers and keywords:
                            for i, keyword in enumerate(keywords):
                                if i < len(numbers):
                                    results[keyword] = int(numbers[i])

                            results["total"] = sum([results["passed"], results["failed"], results["skipped"]])
                            if results["total"] > 0:
                                break

        return results
    
    def _extract_coverage_info(self, stdout: str) -> Optional[float]:
        """커버리지 정보 추출"""
        lines = stdout.split('\n')
        for line in lines:
            # Pattern 1: "TOTAL     1234    567    54%"
            if "TOTAL" in line and "%" in line:
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        try:
                            return float(part.rstrip('%'))
                        except ValueError:
                            continue

            # Pattern 2: Coverage report 마지막 줄 패턴
            # "Required test coverage of 15% reached. Total coverage: 25.43%"
            if "Total coverage:" in line:
                import re
                match = re.search(r'Total coverage:\s*([\d.]+)%', line)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        continue

        return None
    
    def run_group(self, group: Dict[str, Any], verbose: bool = True) -> TestResult:
        """단일 그룹 실행"""
        self.logger.info(f"🚀 Starting Group {group['id']}: {group['name']}")
        
        # 경로 존재 확인
        existing_paths = []
        for path in group["paths"]:
            full_path = self.project_root / path
            if full_path.exists():
                existing_paths.append(path)
            else:
                self.logger.warning(f"⚠️  Path not found: {path}")
        
        if not existing_paths:
            self.logger.error(f"❌ No valid paths found for Group {group['id']}")
            return TestResult(
                group_id=group['id'],
                group_name=group['name'],
                phase=group['phase'],
                paths=group['paths'],
                command="",
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration_seconds=0.0,
                total_tests=0, passed=0, failed=0, skipped=0,
                error_rate=100.0, skip_rate=0.0,
                exit_code=-1
            )
        
        # 명령어 생성 (실제 존재하는 경로만 사용)
        group_copy = group.copy()
        group_copy["paths"] = existing_paths
        command = self._build_pytest_command(group_copy, verbose)
        
        self.logger.info(f"📋 Command: {command}")
        
        # 실행
        start_time = time.time()
        start_time_iso = datetime.now().isoformat()
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=group.get("timeout", 300)
            )
            
            end_time = time.time()
            end_time_iso = datetime.now().isoformat()
            duration = end_time - start_time
            
            # 결과 파싱
            test_results = self._parse_pytest_output(result.stdout, result.stderr)
            coverage = self._extract_coverage_info(result.stdout)
            
            # 메트릭 계산
            total_tests = test_results["total"]
            error_rate = (test_results["failed"] / total_tests * 100) if total_tests > 0 else 0.0
            skip_rate = (test_results["skipped"] / total_tests * 100) if total_tests > 0 else 0.0
            
            # 커버리지 델타 계산
            coverage_delta = None
            if coverage is not None:
                coverage_delta = coverage - self.cumulative_coverage
                self.cumulative_coverage = coverage
            
            self.logger.info(f"✅ Group {group['id']} Complete: "
                           f"{test_results['passed']} passed, "
                           f"{test_results['failed']} failed, "
                           f"{test_results['skipped']} skipped "
                           f"({duration:.1f}s)")
            
            if coverage:
                self.logger.info(f"📊 Coverage: {coverage:.1f}% (+{coverage_delta:.1f}%)")
            
            return TestResult(
                group_id=group['id'],
                group_name=group['name'],
                phase=group['phase'],
                paths=existing_paths,
                command=command,
                start_time=start_time_iso,
                end_time=end_time_iso,
                duration_seconds=duration,
                total_tests=total_tests,
                passed=test_results["passed"],
                failed=test_results["failed"],
                skipped=test_results["skipped"],
                error_rate=error_rate,
                skip_rate=skip_rate,
                coverage_delta=coverage_delta,
                exit_code=result.returncode,
                stdout=result.stdout[-2000:],  # 마지막 2000자만 저장
                stderr=result.stderr[-1000:] if result.stderr else ""
            )
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"⏰ Group {group['id']} timed out after {group.get('timeout', 300)}s")
            return TestResult(
                group_id=group['id'],
                group_name=group['name'],
                phase=group['phase'],
                paths=existing_paths,
                command=command,
                start_time=start_time_iso,
                end_time=datetime.now().isoformat(),
                duration_seconds=group.get("timeout", 300),
                total_tests=0, passed=0, failed=0, skipped=0,
                error_rate=100.0, skip_rate=0.0,
                exit_code=-2,
                stderr="TIMEOUT"
            )
            
        except Exception as e:
            self.logger.error(f"💥 Group {group['id']} failed with exception: {str(e)}")
            return TestResult(
                group_id=group['id'],
                group_name=group['name'],
                phase=group['phase'],
                paths=existing_paths,
                command=command,
                start_time=start_time_iso,
                end_time=datetime.now().isoformat(),
                duration_seconds=0.0,
                total_tests=0, passed=0, failed=0, skipped=0,
                error_rate=100.0, skip_rate=0.0,
                exit_code=-3,
                stderr=str(e)
            )
    
    def run_phase(self, phase_id: int, early_stop: bool = True) -> PhaseResult:
        """Phase 실행"""
        phase_groups = [g for g in self.test_groups if g["phase"] == phase_id]
        phase_name = f"Phase {phase_id}"
        
        if phase_id == 1:
            phase_name += ": 빠른 단위 테스트"
        elif phase_id == 2:
            phase_name += ": 중간 단위 테스트"
        elif phase_id == 3:
            phase_name += ": 통합 및 E2E 테스트"
            
        self.logger.info(f"🎯 Starting {phase_name} ({len(phase_groups)} groups)")
        
        phase_start_time = time.time()
        phase_results = []
        
        for group in phase_groups:
            result = self.run_group(group)
            phase_results.append(result)
            self.results.append(result)
            
            # 조기 중단 검사
            if early_stop and result.error_rate > 50.0:
                self.logger.warning(f"⚠️  Group {group['id']} has high error rate ({result.error_rate:.1f}%), "
                                  f"but continuing...")
        
        phase_duration = time.time() - phase_start_time
        
        # Phase 통계 계산
        total_tests = sum(r.total_tests for r in phase_results)
        total_passed = sum(r.passed for r in phase_results)
        total_failed = sum(r.failed for r in phase_results)
        total_skipped = sum(r.skipped for r in phase_results)
        
        phase_error_rate = (total_failed / total_tests * 100) if total_tests > 0 else 0.0
        phase_skip_rate = (total_skipped / total_tests * 100) if total_tests > 0 else 0.0
        
        self.logger.info(f"🏁 {phase_name} Complete: "
                        f"{total_passed} passed, {total_failed} failed, {total_skipped} skipped "
                        f"({phase_duration:.1f}s)")
        
        phase_result = PhaseResult(
            phase_id=phase_id,
            phase_name=phase_name,
            groups=phase_results,
            total_duration=phase_duration,
            total_tests=total_tests,
            total_passed=total_passed,
            total_failed=total_failed,
            total_skipped=total_skipped,
            phase_error_rate=phase_error_rate,
            phase_skip_rate=phase_skip_rate
        )
        
        self.phase_results.append(phase_result)
        return phase_result
    
    def run_all_phases(self, phases: Optional[List[int]] = None, verbose: bool = True) -> Dict[str, Any]:
        """모든 Phase 실행"""
        if phases is None:
            phases = [1, 2, 3]
            
        self.logger.info(f"🚀 Starting Comprehensive Test Execution (Phases: {phases})")
        execution_start_time = time.time()
        
        for phase_id in phases:
            try:
                self.run_phase(phase_id, early_stop=True)
                self._save_intermediate_results()
            except KeyboardInterrupt:
                self.logger.warning("⛔ Execution interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"💥 Phase {phase_id} failed: {str(e)}")
                if phase_id == 3:  # Phase 3 실패는 계속 진행
                    continue
                else:
                    break
        
        total_duration = time.time() - execution_start_time
        
        # 최종 통계
        final_report = self._generate_final_report(total_duration)
        self._save_final_results(final_report)
        
        self.logger.info(f"🎉 All phases completed in {total_duration/60:.1f} minutes")
        
        return final_report
    
    def _generate_final_report(self, total_duration: float) -> Dict[str, Any]:
        """최종 리포트 생성"""
        total_tests = sum(r.total_tests for r in self.results)
        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        
        # 성능 메트릭
        durations = [r.duration_seconds for r in self.results if r.duration_seconds > 0]
        fastest_group = min(self.results, key=lambda r: r.duration_seconds) if self.results else None
        slowest_group = max(self.results, key=lambda r: r.duration_seconds) if self.results else None
        
        return {
            "execution_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_duration": f"{total_duration/60:.1f}m",
                "total_duration_seconds": total_duration,
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "skipped": total_skipped,
                "overall_error_rate": (total_failed / total_tests * 100) if total_tests > 0 else 0.0,
                "overall_skip_rate": (total_skipped / total_tests * 100) if total_tests > 0 else 0.0,
                "final_coverage": self.cumulative_coverage
            },
            "phase_results": [asdict(pr) for pr in self.phase_results],
            "group_results": [asdict(r) for r in self.results],
            "coverage_progression": [
                {
                    "group": f"Group {r.group_id}",
                    "group_name": r.group_name,
                    "coverage_delta": r.coverage_delta or 0.0,
                    "cumulative_coverage": (r.coverage_delta or 0.0) + 
                        sum(prev.coverage_delta or 0.0 for prev in self.results[:self.results.index(r)])
                }
                for r in self.results if r.coverage_delta is not None
            ],
            "performance_metrics": {
                "fastest_group": fastest_group.group_name if fastest_group else None,
                "fastest_duration": fastest_group.duration_seconds if fastest_group else None,
                "slowest_group": slowest_group.group_name if slowest_group else None,
                "slowest_duration": slowest_group.duration_seconds if slowest_group else None,
                "average_test_time": sum(durations) / len(durations) if durations else 0.0,
                "total_groups_executed": len(self.results)
            }
        }
    
    def _save_intermediate_results(self):
        """중간 결과 저장"""
        intermediate_data = {
            "timestamp": datetime.now().isoformat(),
            "completed_groups": len(self.results),
            "current_coverage": self.cumulative_coverage,
            "results": [asdict(r) for r in self.results]
        }
        
        intermediate_file = self.project_root / "test_metrics_intermediate.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_data, f, indent=2, ensure_ascii=False)
    
    def _save_final_results(self, report: Dict[str, Any]):
        """최종 결과 저장"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📄 Final report saved to: {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description="포괄적 테스트 실행 도구")
    parser.add_argument("--phases", "-p", nargs="+", type=int, default=[1, 2, 3],
                       help="실행할 Phase (기본: 1 2 3)")
    parser.add_argument("--output", "-o", type=Path,
                       help="출력 파일 경로")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="상세 출력")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                       help="프로젝트 루트 경로")
    
    args = parser.parse_args()
    
    # 실행
    runner = ComprehensiveTestRunner(
        project_root=args.project_root,
        output_file=args.output
    )
    
    try:
        final_report = runner.run_all_phases(
            phases=args.phases,
            verbose=args.verbose
        )
        
        # 간단한 요약 출력
        summary = final_report["execution_summary"]
        print(f"\n🎯 최종 결과 요약:")
        print(f"   총 실행시간: {summary['total_duration']}")
        print(f"   총 테스트: {summary['total_tests']}")
        print(f"   성공: {summary['passed']} ({summary['passed']/summary['total_tests']*100:.1f}%)")
        print(f"   실패: {summary['failed']} ({summary['overall_error_rate']:.1f}%)")
        print(f"   스킵: {summary['skipped']} ({summary['overall_skip_rate']:.1f}%)")
        print(f"   최종 커버리지: {summary['final_coverage']:.1f}%")
        print(f"   상세 리포트: {runner.output_file}")
        
        return 0 if summary['failed'] == 0 else 1
        
    except KeyboardInterrupt:
        print("\n⛔ 사용자에 의해 중단되었습니다")
        return 130
    except Exception as e:
        print(f"\n💥 실행 중 오류 발생: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())