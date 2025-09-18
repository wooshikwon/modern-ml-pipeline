#!/usr/bin/env python3
"""
테스트 성능 분석 스크립트

이 스크립트는 다음 기능을 제공합니다:
1. 각 테스트 파일별 실행 시간 측정
2. 커버리지 분석 (파일별, 라인별)
3. 병목 지점 식별
4. 세부 로그 출력

Usage:
    python scripts/test_analysis.py [options]

Options:
    --timeout=300           # 타임아웃 (초)
    --verbose               # 상세 출력
    --coverage-only         # 커버리지만 분석
    --individual-timing     # 개별 테스트 타이밍
    --output-dir=logs       # 출력 디렉토리
"""

import argparse
import subprocess
import sys
import time
import json
import csv
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TestResult:
    """테스트 결과 정보"""
    name: str
    duration: float
    status: str  # PASSED, FAILED, SKIPPED
    file_path: str
    line_number: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class CoverageInfo:
    """커버리지 정보"""
    file_path: str
    total_lines: int
    covered_lines: int
    missing_lines: List[int]
    coverage_percent: float


class TestAnalyzer:
    """테스트 성능 및 커버리지 분석기"""

    def __init__(self, output_dir: str = "logs", timeout: int = 300, verbose: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timeout = timeout
        self.verbose = verbose

        # 로깅 설정
        log_file = self.output_dir / f"test_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.test_results: List[TestResult] = []
        self.coverage_info: Dict[str, CoverageInfo] = {}

    def run_individual_test_timing(self) -> Dict[str, float]:
        """개별 테스트 파일별 실행 시간 측정"""
        self.logger.info("🔍 개별 테스트 파일 타이밍 분석 시작...")

        # 모든 테스트 파일 찾기
        test_files = list(Path("tests").rglob("test_*.py"))
        test_timings = {}

        for test_file in test_files:
            try:
                relative_path = str(test_file.relative_to(Path.cwd()))
            except ValueError:
                # Handle case where test_file is not relative to current directory
                relative_path = str(test_file)
            self.logger.info(f"  📝 테스트 파일: {relative_path}")

            start_time = time.time()

            try:
                # 개별 테스트 파일 실행
                cmd = [
                    "uv", "run", "pytest",
                    str(test_file),
                    "-v",
                    "--tb=short",
                    "--disable-warnings",
                    "--maxfail=5",
                    f"--timeout={min(60, self.timeout // len(test_files))}"  # 파일당 최대 60초 또는 전체 타임아웃의 분할
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout // len(test_files),
                    cwd=Path.cwd()
                )

                end_time = time.time()
                duration = end_time - start_time
                test_timings[relative_path] = duration

                # 결과 파싱
                self._parse_pytest_output(result.stdout, result.stderr, relative_path)

                if result.returncode == 0:
                    self.logger.info(f"    ✅ 성공 ({duration:.2f}초)")
                else:
                    self.logger.warning(f"    ❌ 실패 ({duration:.2f}초)")
                    if self.verbose:
                        self.logger.debug(f"STDOUT: {result.stdout}")
                        self.logger.debug(f"STDERR: {result.stderr}")

            except subprocess.TimeoutExpired:
                end_time = time.time()
                duration = end_time - start_time
                test_timings[relative_path] = duration
                self.logger.error(f"    ⏰ 타임아웃 ({duration:.2f}초)")

            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                test_timings[relative_path] = duration
                self.logger.error(f"    💥 오류: {e} ({duration:.2f}초)")

        # 결과 정렬 (느린 순서대로)
        sorted_timings = dict(sorted(test_timings.items(), key=lambda x: x[1], reverse=True))

        # 타이밍 결과 저장
        timing_file = self.output_dir / "individual_test_timings.json"
        with open(timing_file, 'w') as f:
            json.dump(sorted_timings, f, indent=2)

        self.logger.info(f"📊 개별 테스트 타이밍 저장: {timing_file}")

        # 상위 10개 느린 테스트 출력
        self.logger.info("\n🐌 가장 느린 테스트 파일들 (상위 10개):")
        for i, (file_path, duration) in enumerate(list(sorted_timings.items())[:10], 1):
            self.logger.info(f"  {i:2d}. {duration:8.2f}초 - {file_path}")

        return sorted_timings

    def _parse_pytest_output(self, stdout: str, stderr: str, file_path: str):
        """pytest 출력 파싱하여 개별 테스트 결과 추출"""
        lines = stdout.split('\n')

        for line in lines:
            # 개별 테스트 결과 파싱: "test_file.py::test_function PASSED [100%]"
            if '::' in line and any(status in line for status in ['PASSED', 'FAILED', 'SKIPPED']):
                try:
                    # 정규식으로 파싱
                    match = re.match(r'([^:]+)::([^:\s]+).*?(PASSED|FAILED|SKIPPED)', line)
                    if match:
                        test_file, test_name, status = match.groups()

                        # 실행 시간 추출 (있는 경우)
                        duration_match = re.search(r'\[(\d+\.\d+)s\]', line)
                        duration = float(duration_match.group(1)) if duration_match else 0.0

                        self.test_results.append(TestResult(
                            name=f"{test_file}::{test_name}",
                            duration=duration,
                            status=status,
                            file_path=file_path
                        ))
                except Exception as e:
                    self.logger.debug(f"테스트 결과 파싱 오류: {e} - {line}")

    def run_coverage_analysis(self) -> Dict[str, CoverageInfo]:
        """커버리지 분석 실행"""
        self.logger.info("📈 커버리지 분석 시작...")

        try:
            # 커버리지와 함께 전체 테스트 실행
            cmd = [
                "uv", "run", "pytest",
                "tests/",
                "--cov=src",
                "--cov-report=json",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "--cov-config=.coveragerc",
                "--tb=short",
                "--disable-warnings",
                f"--timeout={self.timeout}"
            ]

            self.logger.info(f"실행 명령어: {' '.join(cmd)}")

            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path.cwd()
            )
            end_time = time.time()

            self.logger.info(f"커버리지 테스트 완료 ({end_time - start_time:.2f}초)")

            if result.returncode != 0:
                self.logger.warning("커버리지 테스트에서 일부 실패가 있었습니다.")
                if self.verbose:
                    self.logger.debug(f"STDOUT: {result.stdout}")
                    self.logger.debug(f"STDERR: {result.stderr}")

            # 커버리지 JSON 결과 파싱
            coverage_json_file = Path("coverage.json")
            if coverage_json_file.exists():
                self._parse_coverage_json(coverage_json_file)
            else:
                self.logger.warning("coverage.json 파일을 찾을 수 없습니다.")

            # 터미널 출력에서 커버리지 정보 추출
            self._parse_coverage_terminal_output(result.stdout)

        except subprocess.TimeoutExpired:
            self.logger.error(f"커버리지 분석이 {self.timeout}초 타임아웃되었습니다.")
        except Exception as e:
            self.logger.error(f"커버리지 분석 중 오류 발생: {e}")

        return self.coverage_info

    def _parse_coverage_json(self, json_file: Path):
        """coverage.json 파일 파싱"""
        try:
            with open(json_file, 'r') as f:
                coverage_data = json.load(f)

            files_data = coverage_data.get('files', {})

            for file_path, file_data in files_data.items():
                summary = file_data.get('summary', {})
                missing_lines = file_data.get('missing_lines', [])

                self.coverage_info[file_path] = CoverageInfo(
                    file_path=file_path,
                    total_lines=summary.get('num_statements', 0),
                    covered_lines=summary.get('covered_lines', 0),
                    missing_lines=missing_lines,
                    coverage_percent=summary.get('percent_covered', 0.0)
                )

        except Exception as e:
            self.logger.error(f"coverage.json 파싱 오류: {e}")

    def _parse_coverage_terminal_output(self, output: str):
        """터미널 커버리지 출력 파싱"""
        lines = output.split('\n')

        # 커버리지 테이블 찾기
        in_coverage_table = False
        for line in lines:
            if '-----' in line and 'coverage' in line.lower():
                in_coverage_table = True
                continue

            if in_coverage_table and line.strip():
                # 커버리지 라인 파싱: "src/components/adapters.py    45    12    73%   23-45, 67-89"
                parts = line.split()
                if len(parts) >= 4 and parts[0].endswith('.py'):
                    try:
                        file_path = parts[0]
                        total_lines = int(parts[1])
                        missing_count = int(parts[2])
                        coverage_percent = float(parts[3].rstrip('%'))

                        covered_lines = total_lines - missing_count

                        # 누락된 라인 범위 파싱
                        missing_lines = []
                        if len(parts) > 4:
                            missing_ranges = parts[4]
                            missing_lines = self._parse_missing_line_ranges(missing_ranges)

                        # 이미 JSON에서 파싱된 정보가 없는 경우에만 추가
                        if file_path not in self.coverage_info:
                            self.coverage_info[file_path] = CoverageInfo(
                                file_path=file_path,
                                total_lines=total_lines,
                                covered_lines=covered_lines,
                                missing_lines=missing_lines,
                                coverage_percent=coverage_percent
                            )
                    except (ValueError, IndexError) as e:
                        self.logger.debug(f"커버리지 라인 파싱 실패: {e} - {line}")

            if in_coverage_table and line.startswith('='):
                break

    def _parse_missing_line_ranges(self, ranges_str: str) -> List[int]:
        """누락된 라인 범위 문자열을 라인 번호 리스트로 변환"""
        missing_lines = []
        ranges = ranges_str.split(', ')

        for range_part in ranges:
            if '-' in range_part:
                # 범위: "23-45"
                start, end = map(int, range_part.split('-'))
                missing_lines.extend(range(start, end + 1))
            else:
                # 단일 라인: "67"
                missing_lines.append(int(range_part))

        return missing_lines

    def generate_reports(self):
        """분석 결과 리포트 생성"""
        self.logger.info("📋 분석 리포트 생성 중...")

        # 1. 테스트 타이밍 리포트
        self._generate_timing_report()

        # 2. 커버리지 리포트
        self._generate_coverage_report()

        # 3. 병목 분석 리포트
        self._generate_bottleneck_report()

        # 4. 종합 요약 리포트
        self._generate_summary_report()

    def _generate_timing_report(self):
        """테스트 타이밍 리포트 생성"""
        timing_csv = self.output_dir / "test_timing_analysis.csv"

        with open(timing_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Test Name', 'Duration (seconds)', 'Status', 'File Path'])

            # 느린 순서대로 정렬
            sorted_tests = sorted(self.test_results, key=lambda x: x.duration, reverse=True)

            for test in sorted_tests:
                writer.writerow([test.name, f"{test.duration:.3f}", test.status, test.file_path])

        self.logger.info(f"테스트 타이밍 리포트 저장: {timing_csv}")

    def _generate_coverage_report(self):
        """커버리지 리포트 생성"""
        coverage_csv = self.output_dir / "coverage_analysis.csv"

        with open(coverage_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['File Path', 'Total Lines', 'Covered Lines', 'Coverage %', 'Missing Lines Count', 'Missing Lines'])

            # 커버리지 낮은 순서대로 정렬
            sorted_coverage = sorted(self.coverage_info.values(), key=lambda x: x.coverage_percent)

            for cov in sorted_coverage:
                missing_lines_str = ', '.join(map(str, cov.missing_lines[:10]))  # 처음 10개만
                if len(cov.missing_lines) > 10:
                    missing_lines_str += f" ... (+{len(cov.missing_lines) - 10} more)"

                writer.writerow([
                    cov.file_path,
                    cov.total_lines,
                    cov.covered_lines,
                    f"{cov.coverage_percent:.1f}%",
                    len(cov.missing_lines),
                    missing_lines_str
                ])

        self.logger.info(f"커버리지 리포트 저장: {coverage_csv}")

    def _generate_bottleneck_report(self):
        """병목 분석 리포트 생성"""
        bottleneck_file = self.output_dir / "bottleneck_analysis.txt"

        with open(bottleneck_file, 'w') as f:
            f.write("🐌 테스트 병목 분석 리포트\n")
            f.write("=" * 50 + "\n\n")

            # 가장 느린 테스트들
            f.write("가장 느린 테스트들 (상위 20개):\n")
            f.write("-" * 30 + "\n")

            sorted_tests = sorted(self.test_results, key=lambda x: x.duration, reverse=True)
            for i, test in enumerate(sorted_tests[:20], 1):
                f.write(f"{i:2d}. {test.duration:8.3f}초 - {test.name} ({test.status})\n")

            f.write("\n")

            # 커버리지가 낮은 파일들
            f.write("커버리지가 낮은 파일들 (50% 미만):\n")
            f.write("-" * 30 + "\n")

            low_coverage = [cov for cov in self.coverage_info.values() if cov.coverage_percent < 50.0]
            low_coverage.sort(key=lambda x: x.coverage_percent)

            for cov in low_coverage:
                f.write(f"{cov.coverage_percent:5.1f}% - {cov.file_path} ({cov.covered_lines}/{cov.total_lines} lines)\n")

            if not low_coverage:
                f.write("모든 파일이 50% 이상의 커버리지를 가지고 있습니다! 👍\n")

        self.logger.info(f"병목 분석 리포트 저장: {bottleneck_file}")

    def _generate_summary_report(self):
        """종합 요약 리포트 생성"""
        summary_file = self.output_dir / "analysis_summary.json"

        # 통계 계산
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t.status == 'PASSED'])
        failed_tests = len([t for t in self.test_results if t.status == 'FAILED'])
        skipped_tests = len([t for t in self.test_results if t.status == 'SKIPPED'])

        total_duration = sum(t.duration for t in self.test_results)
        avg_duration = total_duration / total_tests if total_tests > 0 else 0

        total_coverage_lines = sum(cov.total_lines for cov in self.coverage_info.values())
        total_covered_lines = sum(cov.covered_lines for cov in self.coverage_info.values())
        overall_coverage = (total_covered_lines / total_coverage_lines * 100) if total_coverage_lines > 0 else 0

        summary = {
            "timestamp": datetime.now().isoformat(),
            "test_statistics": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "skipped": skipped_tests,
                "pass_rate": f"{(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%"
            },
            "timing_statistics": {
                "total_duration": f"{total_duration:.2f}s",
                "average_duration": f"{avg_duration:.3f}s",
                "slowest_test": self.test_results[0].name if self.test_results else None,
                "slowest_duration": f"{max((t.duration for t in self.test_results), default=0):.3f}s"
            },
            "coverage_statistics": {
                "overall_coverage": f"{overall_coverage:.1f}%",
                "total_lines": total_coverage_lines,
                "covered_lines": total_covered_lines,
                "files_analyzed": len(self.coverage_info),
                "low_coverage_files": len([cov for cov in self.coverage_info.values() if cov.coverage_percent < 50.0])
            },
            "recommendations": self._generate_recommendations()
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # 콘솔에도 요약 출력
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 테스트 분석 요약")
        self.logger.info("=" * 60)
        self.logger.info(f"📈 테스트 현황: {total_tests}개 중 {passed_tests}개 통과 ({passed_tests/total_tests*100:.1f}%)")
        self.logger.info(f"⏱️  총 실행 시간: {total_duration:.2f}초 (평균 {avg_duration:.3f}초/테스트)")
        self.logger.info(f"📋 전체 커버리지: {overall_coverage:.1f}% ({total_covered_lines:,}/{total_coverage_lines:,} 라인)")
        self.logger.info(f"💾 리포트 저장: {summary_file}")

        return summary

    def _generate_recommendations(self) -> List[str]:
        """최적화 권장사항 생성"""
        recommendations = []

        # 느린 테스트 권장사항
        slow_tests = [t for t in self.test_results if t.duration > 5.0]
        if slow_tests:
            recommendations.append(f"🐌 {len(slow_tests)}개의 느린 테스트(5초 이상)를 최적화하세요.")

        # 낮은 커버리지 권장사항
        low_coverage = [cov for cov in self.coverage_info.values() if cov.coverage_percent < 50.0]
        if low_coverage:
            recommendations.append(f"📈 {len(low_coverage)}개 파일의 커버리지가 50% 미만입니다.")

        # 실패한 테스트 권장사항
        failed_tests = [t for t in self.test_results if t.status == 'FAILED']
        if failed_tests:
            recommendations.append(f"❌ {len(failed_tests)}개의 실패한 테스트를 수정하세요.")

        # 전체적인 권장사항
        total_duration = sum(t.duration for t in self.test_results)
        if total_duration > 300:  # 5분 이상
            recommendations.append("⏰ 전체 테스트 실행 시간이 길어서 병렬화를 고려하세요.")

        if not recommendations:
            recommendations.append("✅ 모든 테스트가 양호한 상태입니다!")

        return recommendations


def main():
    parser = argparse.ArgumentParser(description="Modern ML Pipeline 테스트 분석 도구")
    parser.add_argument("--timeout", type=int, default=300, help="전체 타임아웃 (초)")
    parser.add_argument("--verbose", action="store_true", help="상세 출력")
    parser.add_argument("--coverage-only", action="store_true", help="커버리지만 분석")
    parser.add_argument("--individual-timing", action="store_true", help="개별 테스트 타이밍 분석")
    parser.add_argument("--output-dir", default="logs", help="출력 디렉토리")

    args = parser.parse_args()

    analyzer = TestAnalyzer(
        output_dir=args.output_dir,
        timeout=args.timeout,
        verbose=args.verbose
    )

    print("🚀 Modern ML Pipeline 테스트 분석 시작...")
    print(f"⏰ 타임아웃: {args.timeout}초")
    print(f"📁 출력 디렉토리: {args.output_dir}")

    try:
        if args.individual_timing and not args.coverage_only:
            # 개별 테스트 타이밍 분석
            analyzer.run_individual_test_timing()

        if not args.individual_timing or args.coverage_only:
            # 전체 커버리지 분석
            analyzer.run_coverage_analysis()

        # 리포트 생성
        analyzer.generate_reports()

        print("\n✅ 분석 완료! 리포트를 확인하세요.")
        print(f"📂 출력 디렉토리: {Path(args.output_dir).absolute()}")

    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 분석 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()