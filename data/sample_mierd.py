import json
import random

input_path = "rewrite.jsonl"
output_path = "sampled_6_types.jsonl"

samples = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        scenario = data.get("merdcir_intent_scenario")

        # 跳过没有该字段的样本
        if scenario is None:
            continue

        samples.append(data)

random.shuffle(samples)

# 用于记录已经收集到的 scenario
collected = {}

for data in samples:
    if len(collected) >= 6:
        break

    scenario = data["merdcir_intent_scenario"]

    # 每种 scenario 只保留一个样本
    if scenario not in collected:
        collected[scenario] = data

# 写入新的 jsonl 文件
with open(output_path, "w", encoding="utf-8") as f:
    for sample in collected.values():
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"已保存 {len(collected)} 个不同 scenario 的样本到 {output_path}")
