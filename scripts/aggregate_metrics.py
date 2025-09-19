#!/usr/bin/env python3
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def extract_from_summary(summary: dict) -> dict:
    # Handle multiple schema variants from pytest-json-report
    if not isinstance(summary, dict):
        summary = {}

    # counts might be either in summary directly or nested differently
    total = summary.get("collected", summary.get("total", 0))
    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    skipped = summary.get("skipped", 0)
    errors = summary.get("error", summary.get("errors", 0))
    xfailed = summary.get("xfailed", 0)
    xpassed = summary.get("xpassed", 0)

    # duration may be at summary["duration"] or in top-level durations["total"]
    duration = summary.get("duration", 0.0)

    return {
        "total": int(total) if isinstance(total, (int, float)) else 0,
        "passed": int(passed) if isinstance(passed, (int, float)) else 0,
        "failed": int(failed) if isinstance(failed, (int, float)) else 0,
        "skipped": int(skipped) if isinstance(skipped, (int, float)) else 0,
        "errors": int(errors) if isinstance(errors, (int, float)) else 0,
        "xfailed": int(xfailed) if isinstance(xfailed, (int, float)) else 0,
        "xpassed": int(xpassed) if isinstance(xpassed, (int, float)) else 0,
        "duration_sec": float(duration) if isinstance(duration, (int, float)) else 0.0,
    }


def extract_metrics(report_path: Path) -> dict:
    data = read_json(report_path)
    summary = data.get("summary", {})
    durations = data.get("durations", {})
    metrics = extract_from_summary(summary)

    # Fallback duration from durations["total"] if summary lacked duration
    if metrics.get("duration_sec", 0.0) == 0.0 and isinstance(durations, dict):
        total_dur = durations.get("total")
        if isinstance(total_dur, (int, float)):
            metrics["duration_sec"] = float(total_dur)

    total_tests = metrics.get("total", 0)
    failed = metrics.get("failed", 0) + metrics.get("errors", 0)
    skipped = metrics.get("skipped", 0)

    metrics["fail_rate"] = (failed / total_tests) if total_tests else 0.0
    metrics["skip_rate"] = (skipped / total_tests) if total_tests else 0.0
    return metrics


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)

    groups = ["unit", "integration", "e2e"]
    out = {}
    for g in groups:
        report_path = reports_dir / f"pytest.{g}.json"
        out[g] = extract_metrics(report_path)

    (reports_dir / "metrics.summary.json").write_text(json.dumps(out, indent=2))
    print("Wrote:", reports_dir / "metrics.summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

