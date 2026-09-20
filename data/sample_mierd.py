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

        # Skip records missing the required field.
        if scenario is None:
            continue

        samples.append(data)

random.shuffle(samples)

# Track scenarios that have already been selected.
collected = {}

for data in samples:
    if len(collected) >= 6:
        break

    scenario = data["merdcir_intent_scenario"]

    # Keep one sample per scenario.
    if scenario not in collected:
        collected[scenario] = data

# Write the sampled records to a new JSONL file.
with open(output_path, "w", encoding="utf-8") as f:
    for sample in collected.values():
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Saved {len(collected)} scenario-diverse samples to {output_path}")
