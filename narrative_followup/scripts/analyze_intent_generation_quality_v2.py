#!/usr/bin/env python3
"""Audit the locally available 50K sampled-intent rewrite artifact."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "data/pair_control/train_surfacediverse_raw.jsonl"
OUT_JSON = ROOT / "narrative_followup/metrics/intent_generation_quality_v2.json"
OUT_MD = ROOT / "narrative_followup/reports/intent_generation_quality_v2.md"
SCENARIOS = {
    "surface_diverse_1",
    "surface_diverse_2",
    "surface_diverse_3",
    "surface_diverse_4",
    "surface_diverse_5",
    "surface_diverse_6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    total = json_errors = missing_id = invalid_scenario = 0
    empty_or_nontext_output = 0
    valid_output = 0
    valid_joint = 0
    nps_valid = 0
    scenarios: Counter[str] = Counter()
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                continue
            if not row.get("id"):
                missing_id += 1
            scenario = row.get("merdcir_intent_scenario")
            scenario_valid = scenario in SCENARIOS
            if not scenario_valid:
                invalid_scenario += 1
            else:
                scenarios[scenario] += 1
            output = row.get("merdcir_modification")
            output_valid = isinstance(output, str) and bool(output.strip())
            if output_valid:
                valid_output += 1
            else:
                empty_or_nontext_output += 1
            if scenario_valid and output_valid:
                valid_joint += 1
            if isinstance(row.get("nps"), list):
                nps_valid += 1

    valid_assignment = total - json_errors - invalid_scenario
    valid_rows = sum(scenarios.values())
    payload = {
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "rows_total": total,
        "json_parse_errors": json_errors,
        "missing_id": missing_id,
        "invalid_or_missing_scenario": invalid_scenario,
        "valid_scenario_assignments": valid_assignment,
        "valid_rewrite_output_nonempty_text": valid_output,
        "empty_or_nontext_rewrite_output": empty_or_nontext_output,
        "rows_with_valid_scenario_and_output": valid_joint,
        "nps_field_is_list": nps_valid,
        "scenario_counts": dict(sorted(scenarios.items())),
        "scenario_proportions": {
            key: value / valid_rows if valid_rows else 0.0
            for key, value in sorted(scenarios.items())
        },
        "automatic_only": True,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Intent generation quality v2",
        "",
        "This is a structural audit of the locally available 50K sampled-intent",
        "rewrite artifact. It does not validate semantic correctness of a rewrite",
        "or establish causal intent effects.",
        "",
        "| Check | Count |",
        "|---|---:|",
        f"| total JSONL rows | {total:,} |",
        f"| JSON parse errors | {json_errors:,} |",
        f"| invalid/missing scenario | {invalid_scenario:,} |",
        f"| valid scenario assignments | {valid_assignment:,} |",
        f"| non-empty text rewrite outputs | {valid_output:,} |",
        f"| empty/non-text rewrite outputs | {empty_or_nontext_output:,} |",
        f"| rows with valid scenario and output | {payload['rows_with_valid_scenario_and_output']:,} |",
        f"| rows with list-valued NP field | {nps_valid:,} |",
        "",
        "## Scenario assignments",
        "",
        "| Scenario | Count | Proportion |",
        "|---|---:|---:|",
    ]
    for scenario, count in sorted(scenarios.items()):
        lines.append(f"| {scenario} | {count:,} | {count / valid_rows:.6f} |")
    lines += [
        "",
        f"Source SHA256: `{payload['source_sha256']}`",
        "",
        "The counts above are structural/automatic evidence. Human review is",
        "still required to assess whether the assigned scenario and rewritten",
        "text are semantically valid.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote={OUT_JSON}")
    print(f"wrote={OUT_MD}")
    print(json.dumps({k: payload[k] for k in ('rows_total', 'valid_scenario_assignments', 'valid_rewrite_output_nonempty_text', 'empty_or_nontext_rewrite_output', 'json_parse_errors', 'invalid_or_missing_scenario')}, indent=2))
    print("status=automatic_structural_only; semantic_human_validation_required=yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
