import argparse
import math
from pathlib import Path


# Reproduction use:
#   python split_for.py --input-jsonl data/rewrite.jsonl --num-splits 12
#
# This helper splits a large JSONL file into `part_0.jsonl`, `part_1.jsonl`,
# etc. Generated shards are ignored by git and are meant to be recreated
# locally.


def parse_args():
    parser = argparse.ArgumentParser(description="Split a JSONL file into numbered shards.")
    parser.add_argument("--input-jsonl", default="./data/rewrite.jsonl", help="Source JSONL file.")
    parser.add_argument("--num-splits", type=int, default=12, help="Number of output shards.")
    parser.add_argument("--output-dir", default=".", help="Directory for generated shards.")
    parser.add_argument("--prefix", default="part", help="Output filename prefix.")
    return parser.parse_args()


def split_jsonl(input_file, num_splits, output_dir=".", prefix="part"):
    if num_splits <= 0:
        raise ValueError("num_splits must be positive.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 计算总行数
    print(f"正在统计行数: {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    lines_per_file = math.ceil(total_lines / num_splits)
    print(f"总行数: {total_lines}, 计划分为 {num_splits} 份, 每份约 {lines_per_file} 行。")

    # 2. 开始拆分
    with open(input_file, 'r', encoding='utf-8') as f:
        for i in range(num_splits):
            output_filename = output_dir / f"{prefix}_{i}.jsonl"
            print(f"正在写入: {output_filename}")
            
            with open(output_filename, 'w', encoding='utf-8') as out_f:
                count = 0
                while count < lines_per_file:
                    line = f.readline()
                    if not line:
                        break
                    out_f.write(line)
                    count += 1
            
            if not line: # 文件读完了
                break

    print("拆分完成！")


if __name__ == "__main__":
    args = parse_args()
    split_jsonl(args.input_jsonl, args.num_splits, args.output_dir, args.prefix)
