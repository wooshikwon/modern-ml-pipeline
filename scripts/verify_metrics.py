#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path


def load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics summary not found: {path}")
    return json.loads(path.read_text())


def get_thresholds() -> dict:
    # Defaults; override via env vars if needed
    return {
        "FAIL_RATE_MAX": float(os.getenv("FAIL_RATE_MAX", "0.0")),
        "SKIP_RATE_MAX": float(os.getenv("SKIP_RATE_MAX", "0.15")),
        # Optional duration checks (seconds). Set to 0 or negative to disable.
        "DURATION_MAX_UNIT": float(os.getenv("DURATION_MAX_UNIT", "0")),
        "DURATION_MAX_INTEGRATION": float(os.getenv("DURATION_MAX_INTEGRATION", "0")),
        "DURATION_MAX_E2E": float(os.getenv("DURATION_MAX_E2E", "0")),
    }


def verify_group_metrics(group: str, metrics: dict, thresholds: dict) -> list:
    violations = []
    fail_rate = metrics.get("fail_rate", 0.0)
    skip_rate = metrics.get("skip_rate", 0.0)
    duration = metrics.get("duration_sec", 0.0)

    if fail_rate > thresholds["FAIL_RATE_MAX"]:
        violations.append(
            f"{group}: fail_rate {fail_rate:.3f} > {thresholds['FAIL_RATE_MAX']:.3f}"
        )

    if skip_rate > thresholds["SKIP_RATE_MAX"]:
        violations.append(
            f"{group}: skip_rate {skip_rate:.3f} > {thresholds['SKIP_RATE_MAX']:.3f}"
        )

    # Duration checks (optional)
    if group == "unit" and thresholds["DURATION_MAX_UNIT"] > 0 and duration > thresholds["DURATION_MAX_UNIT"]:
        violations.append(
            f"{group}: duration {duration:.1f}s > {thresholds['DURATION_MAX_UNIT']:.1f}s"
        )
    if group == "integration" and thresholds["DURATION_MAX_INTEGRATION"] > 0 and duration > thresholds["DURATION_MAX_INTEGRATION"]:
        violations.append(
            f"{group}: duration {duration:.1f}s > {thresholds['DURATION_MAX_INTEGRATION']:.1f}s"
        )
    if group == "e2e" and thresholds["DURATION_MAX_E2E"] > 0 and duration > thresholds["DURATION_MAX_E2E"]:
        violations.append(
            f"{group}: duration {duration:.1f}s > {thresholds['DURATION_MAX_E2E']:.1f}s"
        )

    return violations


def verify_coverage_files(reports_dir: Path) -> list:
    violations = []
    for group in ("unit", "integration", "e2e"):
        path = reports_dir / f"coverage.{group}.xml"
        if not path.exists():
            violations.append(f"coverage XML missing for {group}: {path}")
    return violations


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    reports_dir = root / "reports"
    summary_path = reports_dir / "metrics.summary.json"

    thresholds = get_thresholds()
    summary = load_summary(summary_path)

    all_violations = []

    # Verify per-group metrics
    for group in ("unit", "integration", "e2e"):
        group_metrics = summary.get(group, {})
        if not group_metrics:
            all_violations.append(f"missing metrics for group: {group}")
            continue
        all_violations.extend(verify_group_metrics(group, group_metrics, thresholds))

    # Verify coverage files existence
    all_violations.extend(verify_coverage_files(reports_dir))

    # Emit verification report
    verification = {
        "thresholds": thresholds,
        "violations": all_violations,
        "passed": len(all_violations) == 0,
    }
    (reports_dir / "metrics.verification.json").write_text(json.dumps(verification, indent=2))

    if verification["passed"]:
        print("Metrics verification passed.")
        return 0
    else:
        print("Metrics verification failed:\n - " + "\n - ".join(all_violations))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

