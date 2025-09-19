#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path


def _prepare_env_for_group(group_name: str) -> dict:
    env = os.environ.copy()
    # Quiet interactive prompts during tests
    env.setdefault("MMP_QUIET_PROMPTS", "1")
    # Coverage file per group
    env["COVERAGE_FILE"] = f".coverage.{group_name}"

    # Ensure MLflow uses a writable, isolated file store per group (best-effort)
    tmp_dir = Path("reports") / "tmp" / group_name
    (tmp_dir / "mlruns").mkdir(parents=True, exist_ok=True)
    env.setdefault("MLFLOW_TRACKING_URI", f"file://{(tmp_dir / 'mlruns').resolve().as_posix()}")

    # Do not opt-in to global process killing by default
    env.setdefault("MMP_ENABLE_GLOBAL_KILL", "0")
    return env


def _effective_workers(group_name: str, requested: str | None) -> str:
    if requested in (None, "", "auto"):
        requested = "auto"
    # Safer defaults: run serial unless explicitly scaled
    if group_name in ("unit", "integration", "e2e"):
        return os.getenv(f"{group_name.upper()}_WORKERS", "1") if requested == "auto" else requested
    return requested


def run_pytest(group_name: str, test_path: str, parallel: str, cov_xml: Path, json_out: Path) -> dict:
    env = _prepare_env_for_group(group_name)
    effective_parallel = _effective_workers(group_name, parallel)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_path,
        "-q",
        "-rA",
        "--durations=0",
        f"-n={effective_parallel}",
        "--timeout=60",
        "--dist=loadscope",
        "--maxfail=0",
        "--cov=src",
        f"--cov-report=xml:{cov_xml}",
        "--cov-report=term-missing",
        "--json-report",
        f"--json-report-file={json_out}",
    ]
    print("RUN:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, env=env, text=True)
        return {"returncode": proc.returncode}
    except KeyboardInterrupt:
        return {"returncode": 130}


def _safe_int(value, default=0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def extract_metrics(json_report_path: Path) -> dict:
    data = json.loads(json_report_path.read_text())

    # Prefer summary counters if present
    summary = data.get("summary", {})
    tests_list = data.get("tests", []) or []

    passed = failed = skipped = error = xfailed = xpassed = 0
    total_duration = 0.0

    if isinstance(summary, dict) and summary:
        collected_val = summary.get("collected", summary.get("total", 0))
        passed = _safe_int(summary.get("passed", 0))
        failed = _safe_int(summary.get("failed", 0))
        skipped = _safe_int(summary.get("skipped", 0))
        error = _safe_int(summary.get("error", summary.get("errors", 0)))
        xfailed = _safe_int(summary.get("xfailed", 0))
        xpassed = _safe_int(summary.get("xpassed", 0))
        total_tests = _safe_int(collected_val, 0)
    else:
        # Fallback: derive from individual test cases if summary schema is unexpected
        for t in tests_list:
            outcome = t.get("outcome")
            if outcome == "passed":
                passed += 1
            elif outcome == "failed":
                failed += 1
            elif outcome == "skipped":
                skipped += 1
            elif outcome == "error":
                error += 1
            elif outcome == "xfailed":
                xfailed += 1
            elif outcome == "xpassed":
                xpassed += 1
        total_tests = passed + failed + skipped + error + xfailed + xpassed

    # Duration: use top-level duration if present, else try durations.total
    total_duration = data.get("duration")
    if total_duration is None:
        durations = data.get("durations", {}) or {}
        total_duration = durations.get("total", 0.0)
    try:
        total_duration = float(total_duration)
    except Exception:
        total_duration = 0.0

    skip_rate = (skipped / total_tests) if total_tests else 0.0
    fail_rate = ((failed + error) / total_tests) if total_tests else 0.0

    return {
        "total": total_tests,
        "passed": passed,
        "failed": failed,
        "errors": error,
        "skipped": skipped,
        "xfailed": xfailed,
        "xpassed": xpassed,
        "skip_rate": skip_rate,
        "fail_rate": fail_rate,
        "duration_sec": total_duration,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)

    groups = [
        {"name": "unit", "path": "tests/unit", "parallel": os.getenv("UNIT_WORKERS", "1")},
        {"name": "integration", "path": "tests/integration", "parallel": os.getenv("INTEGRATION_WORKERS", "1")},
        {"name": "e2e", "path": "tests/e2e", "parallel": os.getenv("E2E_WORKERS", "1")},
    ]

    all_metrics = {}
    overall_rc = 0

    for g in groups:
        name = g["name"]
        path = g["path"]
        parallel = g["parallel"]

        cov_xml = reports_dir / f"coverage.{name}.xml"
        json_out = reports_dir / f"pytest.{name}.json"

        result = run_pytest(name, path, parallel, cov_xml, json_out)
        overall_rc = overall_rc or result.get("returncode", 1)

        if json_out.exists():
            try:
                all_metrics[name] = extract_metrics(json_out)
            except Exception as e:
                all_metrics[name] = {"error": f"metrics extraction failed: {e}"}
        else:
            all_metrics[name] = {"error": "json report missing", "returncode": result.get("returncode")}

    (reports_dir / "metrics.summary.json").write_text(json.dumps(all_metrics, indent=2))
    print("Metrics summary written to:", reports_dir / "metrics.summary.json")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())

