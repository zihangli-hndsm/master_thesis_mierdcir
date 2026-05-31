import argparse
import random
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    # Reproduction use:
    #   python data/split_merdcir_np.py \
    #       --input-jsonl data/merdcir_np/merged.jsonl \
    #       --output-dir data/merdcir_np \
    #       --eval-size 5000 \
    #       --seed 114514
    #
    # The seed controls the exact train/eval split. Record it in experiment
    # notes if you need to reproduce reported metrics exactly.
    parser = argparse.ArgumentParser(
        description=(
            "Split a JSONL dataset into a random eval subset and a train subset."
        )
    )
    parser.add_argument("--input-jsonl", required=True, help="Path to the source JSONL dataset.")
    parser.add_argument(
        "--output-dir",
        default="./merdcir_np",
        help="Output directory for eval/train JSONL files.",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=5000,
        help="Number of examples in the eval subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=114514,
        help="Random seed for reproducible splitting.",
    )
    return parser.parse_args()


def load_jsonl_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line for line in f if line.strip()]


def write_lines(path: Path, lines: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)

    lines = load_jsonl_lines(input_path)
    total = len(lines)
    if args.eval_size <= 0:
        raise ValueError("--eval-size must be a positive integer.")
    if total <= args.eval_size:
        raise ValueError(
            f"Dataset size ({total}) must be larger than eval size ({args.eval_size})."
        )

    rng = random.Random(args.seed)
    indices = list(range(total))
    rng.shuffle(indices)

    eval_indices = set(indices[: args.eval_size])
    eval_lines = [line for i, line in enumerate(lines) if i in eval_indices]
    train_lines = [line for i, line in enumerate(lines) if i not in eval_indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_path = output_dir / "eval/full_eval.jsonl"
    train_path = output_dir / "full_train.jsonl"

    write_lines(eval_path, eval_lines)
    write_lines(train_path, train_lines)

    print(f"Loaded {total} records from: {input_path}")
    print(f"Wrote eval subset ({len(eval_lines)}): {eval_path}")
    print(f"Wrote train subset ({len(train_lines)}): {train_path}")


if __name__ == "__main__":
    main()
