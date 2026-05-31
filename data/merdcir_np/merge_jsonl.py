import argparse
import pandas as pd
import glob

# Reproduction use:
#   cd data/merdcir_np
#   python merge_jsonl.py --pattern 'part_*.jsonl' --output-jsonl merged.jsonl
#
# Keep generated JSONL files out of git; rerun this after regenerating or
# copying the source shards.


def parse_args():
    parser = argparse.ArgumentParser(description="Merge JSONL shards into one JSONL file.")
    parser.add_argument("--pattern", default="*.jsonl", help="Glob pattern for input shards.")
    parser.add_argument("--output-jsonl", default="merged.jsonl", help="Merged output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    all_files = sorted(f for f in glob.glob(args.pattern) if f != args.output_jsonl)
    if not all_files:
        raise FileNotFoundError(f"No JSONL files matched pattern: {args.pattern}")

    df = pd.concat((pd.read_json(f, lines=True) for f in all_files), ignore_index=True)
    df.to_json(args.output_jsonl, orient="records", lines=True, force_ascii=False)
    print(f"Merged {len(all_files)} files into {args.output_jsonl}")


if __name__ == "__main__":
    main()
