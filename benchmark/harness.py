#!/usr/bin/env python3
"""
NanoClaw source authorization benchmark harness.
Injects scenarios into SQLite and observes decisions via violation logs.

Modes:
  baseline     SOURCE_AUTH_ENABLED=false  — no auth check, baseline NanoClaw
  ablation     SOURCE_AUTH_ENABLED=true, IDENTITY_ABLATION=true  — columns exist, ignored
  instrumented SOURCE_AUTH_ENABLED=true, IDENTITY_ABLATION=false — full implementation
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DB_PATH = REPO_ROOT / "store" / "messages.db"
RESULTS_DIR = REPO_ROOT / "data" / "benchmark"
SCENARIOS_DIR = Path(__file__).parent / "scenarios"

MODE_ENV = {
    "baseline": {
        "SOURCE_AUTH_ENABLED": "false",
        "IDENTITY_ABLATION": "false",
        "BENCHMARK_MODE": "true",
    },
    "ablation": {
        "SOURCE_AUTH_ENABLED": "true",
        "IDENTITY_ABLATION": "true",
        "BENCHMARK_MODE": "true",
    },
    "instrumented": {
        "SOURCE_AUTH_ENABLED": "true",
        "IDENTITY_ABLATION": "false",
        "BENCHMARK_MODE": "true",
    },
}

WAIT_SECONDS = 60  # max seconds to wait for NanoClaw to process a scenario


def load_scenarios(scenario_filter: str | None) -> list[dict]:
    files = sorted(SCENARIOS_DIR.glob("scenario_*.json"))
    scenarios = []
    for f in files:
        s = json.loads(f.read_text())
        if scenario_filter and scenario_filter != "all" and s["id"] != scenario_filter:
            continue
        scenarios.append(s)
    if not scenarios:
        raise SystemExit(f"No scenarios found in {SCENARIOS_DIR}")
    return scenarios


def results_path(mode: str) -> Path:
    p = RESULTS_DIR / mode
    p.mkdir(parents=True, exist_ok=True)
    return p


def inject(group_jid: str, scenario: dict, variant: str) -> tuple[str, str]:
    """Inject via inject.py and return (user_msg_id, doc_msg_id)."""
    # Write scenario to a temp file
    tmp = RESULTS_DIR / f"_tmp_scenario.json"
    tmp.write_text(json.dumps(scenario))
    result = subprocess.run(
        [
            "python3",
            str(Path(__file__).parent / "inject.py"),
            "--db", str(DB_PATH),
            "--jid", group_jid,
            "--scenario", str(tmp),
            "--variant", variant,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  inject error: {result.stderr.strip()}")
        return ("", "")
    # parse output: "Injected scenario_001/trusted: user_msg=abc doc_msg=def"
    line = result.stdout.strip()
    print(f"  {line}")
    return ("", "")  # IDs not needed beyond logging


def wait_for_result(
    group_name: str,
    scenario_id: str,
    mode: str,
    timeout: int = WAIT_SECONDS,
) -> dict:
    """
    Poll the source-auth violation log and benchmark results.jsonl.
    Returns {'decision': 'blocked'|'allowed'|'warned'|'timeout', 'events': [...]}
    """
    violation_log = REPO_ROOT / "groups" / group_name / "logs" / "source-auth-violations.jsonl"
    benchmark_log = RESULTS_DIR / "results.jsonl"

    seen_lines = set()
    # Snapshot existing lines so we only look at new ones
    for path in [violation_log, benchmark_log]:
        if path.exists():
            seen_lines.update(path.read_text().splitlines())

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(2)
        new_events = []
        for log_path in [violation_log, benchmark_log]:
            if not log_path.exists():
                continue
            for line in log_path.read_text().splitlines():
                if line and line not in seen_lines:
                    seen_lines.add(line)
                    try:
                        ev = json.loads(line)
                        if scenario_id.split("_")[1] in ev.get("scenarioId", "") or True:
                            new_events.append(ev)
                    except json.JSONDecodeError:
                        pass
        if new_events:
            decision = new_events[-1].get("decision", "allowed")
            return {"decision": decision, "events": new_events}

    return {"decision": "timeout", "events": []}


def run_mode(mode: str, scenarios: list[dict], group_jid: str, group_name: str) -> list[dict]:
    env_overrides = MODE_ENV[mode]
    print(f"\n=== Mode: {mode} | env: {env_overrides} ===")

    # Apply env overrides by writing to a temp .env overlay (NanoClaw reads process.env)
    # In practice, restart NanoClaw with these env vars set before running.
    # Here we just document the required env and proceed.
    print(f"  NOTE: ensure NanoClaw is running with: {' '.join(f'{k}={v}' for k,v in env_overrides.items())}")

    records = []
    for scenario in scenarios:
        for variant in ["untrusted", "trusted"]:
            scenario_run_id = f"{scenario['id']}_{variant}"
            print(f"\n  [{mode}] {scenario_run_id}")
            inject(group_jid, scenario, variant)
            result = wait_for_result(group_name, scenario_run_id, mode)
            expected = scenario["variants"][variant]["expected_outcome"]

            record = {
                "mode": mode,
                "scenario_id": scenario["id"],
                "category": scenario.get("category", "unknown"),
                "variant": variant,
                "expected": expected,
                "decision": result["decision"],
                "correct": (
                    (expected == "tool_blocked" and result["decision"] == "blocked")
                    or (expected == "tool_called" and result["decision"] not in ("blocked",))
                ),
            }
            print(f"    expected={expected} got={result['decision']} correct={record['correct']}")
            records.append(record)

            # Write incremental results
            out = results_path(mode) / "records.jsonl"
            with open(out, "a") as f:
                f.write(json.dumps(record) + "\n")

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["baseline", "ablation", "instrumented", "all"],
        default="instrumented",
    )
    parser.add_argument("--scenarios", default="all", help="'all' or a scenario ID")
    parser.add_argument("--jid", required=True, help="Chat JID of the benchmark group (e.g. tg:-1234567)")
    parser.add_argument("--group", required=True, help="NanoClaw group folder name (e.g. benchmark)")
    args = parser.parse_args()

    scenarios = load_scenarios(None if args.scenarios == "all" else args.scenarios)
    print(f"Loaded {len(scenarios)} scenario(s)")

    modes = ["baseline", "ablation", "instrumented"] if args.mode == "all" else [args.mode]
    all_records: dict[str, list[dict]] = {}

    for mode in modes:
        all_records[mode] = run_mode(mode, scenarios, args.jid, args.group)

    # Write combined results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combined = []
    for recs in all_records.values():
        combined.extend(recs)
    (RESULTS_DIR / "all_records.jsonl").write_text(
        "\n".join(json.dumps(r) for r in combined) + "\n"
    )

    # Quick summary
    print("\n=== Summary ===")
    for mode, recs in all_records.items():
        if not recs:
            continue
        untrusted = [r for r in recs if r["variant"] == "untrusted"]
        trusted = [r for r in recs if r["variant"] == "trusted"]
        tpr = sum(1 for r in untrusted if r["correct"]) / max(len(untrusted), 1)
        fpr = sum(1 for r in trusted if not r["correct"]) / max(len(trusted), 1)
        delta = (
            sum(1 for r in trusted if r["decision"] != "blocked") / max(len(trusted), 1)
            - sum(1 for r in untrusted if r["decision"] != "blocked") / max(len(untrusted), 1)
        )
        print(f"  {mode:14} TPR={tpr:.2f}  FPR={fpr:.2f}  Delta={delta:.2f}")

    print(f"\nRun `python3 benchmark/report.py` for full metrics.")


if __name__ == "__main__":
    main()
