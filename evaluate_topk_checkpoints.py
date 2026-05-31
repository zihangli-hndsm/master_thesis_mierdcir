import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


CHECKPOINT_ROOT = Path("checkpoints")
SUMMARY_PATH = Path("checkpoint_eval_results.md")
METRIC_DATASETS = ("MTCIR", "MerdCIR", "FashionIQ")
CIRR_EXPORT_METRICS = ("recall", "recall_subset")
SCORE_RE = re.compile(r"_score_([0-9.]+)\.pth\.tar$")
METRIC_RE = re.compile(r"^(Recall(?:_subset)?@\d+|mAP):\s*([-+0-9.eE]+)\s*$")


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def checkpoint_score(path):
    match = SCORE_RE.search(path.name)
    if not match:
        return float("-inf")
    return float(match.group(1))


def topk_folders(root):
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("topk"))


def checkpoint_paths(folder):
    return sorted(folder.glob("*.pth.tar"), key=lambda path: (-checkpoint_score(path), path.name))


def infer_model_method(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    keys = state_dict.keys()
    if any(key.startswith("alpha_gen.cross_attn.") for key in keys):
        return "cross_pooling_alpha"
    if any(key.startswith("alpha_gen.") for key in keys):
        return "cross_attn_alpha"
    return "cross_attn"


def result_stem(checkpoint_path):
    return checkpoint_path.name.removesuffix(".pth.tar")


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_metric_stdout(stdout):
    metrics = {}
    for line in stdout.splitlines():
        match = METRIC_RE.match(line.strip())
        if match:
            metrics[match.group(1)] = float(match.group(2))
    return metrics


def run_eval(args, output_path=None):
    cmd = [sys.executable, "eval_checkpoint.py", *args]
    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    payload = {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "status": "ok" if completed.returncode == 0 else "error",
    }
    if output_path is not None and completed.returncode == 0:
        payload["output_json"] = str(output_path)
    if completed.returncode != 0:
        payload["error"] = completed.stdout.strip().splitlines()[-20:]
    return payload


def evaluate_metric_dataset(checkpoint_path, method, dataset, output_path, force):
    if output_path.exists() and not force:
        cached = read_json(output_path)
        if cached.get("status") == "ok":
            return cached

    payload = run_eval([
        "--checkpoint", str(checkpoint_path),
        "--dataset", dataset,
        "--method", method,
    ])
    payload.update({
        "checkpoint": str(checkpoint_path),
        "dataset": dataset,
        "method": method,
        "metrics": parse_metric_stdout(payload["stdout"]),
        "created_at": utc_now(),
    })
    write_json(output_path, payload)
    return payload


def evaluate_cirr_export(checkpoint_path, method, cirr_metric, output_path, metadata_path, force):
    if output_path.exists() and metadata_path.exists() and not force:
        return read_json(metadata_path)

    payload = run_eval([
        "--checkpoint", str(checkpoint_path),
        "--dataset", "CIRR",
        "--method", method,
        "--cirr-metric", cirr_metric,
        "--output-json", str(output_path),
    ], output_path=output_path)
    payload.update({
        "checkpoint": str(checkpoint_path),
        "dataset": "CIRR",
        "cirr_metric": cirr_metric,
        "method": method,
        "created_at": utc_now(),
    })
    write_json(metadata_path, payload)
    return payload


def format_metric(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return ""


def markdown_link(path):
    return f"[`{path.name}`]({path.as_posix()})"


def write_markdown(path, rows, started_at, finished_at):
    metric_keys = ["Recall@1", "Recall@5", "Recall@10", "Recall@50", "mAP"]
    lines = [
        "# Top-k Checkpoint Evaluation Results",
        "",
        f"- Started: `{started_at}`",
        f"- Finished: `{finished_at}`",
        "- Scope: every `.pth.tar` checkpoint in every `checkpoints/topk*` folder.",
        "- Metric datasets: `MTCIR`, `MerdCIR`, `FashionIQ` val split.",
        "- CIRR: exported `recall` and `recall_subset` prediction JSON files into the corresponding top-k folder.",
        "",
        "| Folder | Checkpoint | Score | Model method | Dataset | "
        + " | ".join(metric_keys)
        + " | CIRR JSON | Status |",
        "|---|---|---:|---|---|" + "|".join(["---:"] * len(metric_keys)) + "|---|---|",
    ]

    for row in rows:
        metrics = row.get("metrics") or {}
        metric_values = [format_metric(metrics.get(key)) for key in metric_keys]
        cirr_json = row.get("cirr_json", "")
        if cirr_json:
            cirr_json = markdown_link(Path(cirr_json))
        lines.append(
            "| {folder} | `{checkpoint}` | {score:.6f} | `{method}` | {dataset} | {metrics} | {cirr_json} | {status} |".format(
                folder=row["folder"],
                checkpoint=row["checkpoint"],
                score=row["score"],
                method=row["method"],
                dataset=row["dataset"],
                metrics=" | ".join(metric_values),
                cirr_json=cirr_json,
                status=row["status"],
            )
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate every checkpoint under checkpoints/topk* folders.")
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--force", action="store_true", help="Rerun evaluations even when output JSON exists.")
    args = parser.parse_args()

    started_at = utc_now()
    rows = []
    folders = topk_folders(args.checkpoint_root)
    if not folders:
        raise FileNotFoundError(f"No topk folders found under {args.checkpoint_root}")

    for folder in folders:
        checkpoints = checkpoint_paths(folder)
        if not checkpoints:
            print(f"Skipping {folder}: no checkpoint files.", flush=True)
            continue
        folder_rows = []
        for checkpoint_path in checkpoints:
            score = checkpoint_score(checkpoint_path)
            method = infer_model_method(checkpoint_path)
            stem = result_stem(checkpoint_path)
            print(f"\n=== {folder.name}/{checkpoint_path.name} ({method}) ===", flush=True)

            for dataset in METRIC_DATASETS:
                output_path = folder / f"{stem}__{dataset.lower()}_metrics.json"
                print(f"Evaluating {dataset}...", flush=True)
                result = evaluate_metric_dataset(checkpoint_path, method, dataset, output_path, args.force)
                row = {
                    "folder": folder.name,
                    "checkpoint": checkpoint_path.name,
                    "score": score,
                    "method": method,
                    "dataset": dataset,
                    "metrics": result.get("metrics", {}),
                    "status": result.get("status", "unknown"),
                }
                rows.append(row)
                folder_rows.append(row)

            for cirr_metric in CIRR_EXPORT_METRICS:
                output_path = folder / f"{stem}__cirr_{cirr_metric}.json"
                metadata_path = folder / f"{stem}__cirr_{cirr_metric}_run.json"
                print(f"Exporting CIRR {cirr_metric} JSON...", flush=True)
                result = evaluate_cirr_export(
                    checkpoint_path,
                    method,
                    cirr_metric,
                    output_path,
                    metadata_path,
                    args.force,
                )
                row = {
                    "folder": folder.name,
                    "checkpoint": checkpoint_path.name,
                    "score": score,
                    "method": method,
                    "dataset": f"CIRR/{cirr_metric}",
                    "metrics": {},
                    "cirr_json": str(output_path),
                    "status": result.get("status", "unknown"),
                }
                rows.append(row)
                folder_rows.append(row)

        write_json(folder / "evaluation_results.json", folder_rows)

    finished_at = utc_now()
    write_markdown(args.summary_path, rows, started_at, finished_at)
    print(f"\nWrote {args.summary_path}", flush=True)


if __name__ == "__main__":
    main()
