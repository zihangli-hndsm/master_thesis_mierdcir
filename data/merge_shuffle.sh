#!/bin/bash

PROCESSED=$1
REST=$2
OUTPUT=${3:-merged_shuffled.jsonl}

if [ -z "$PROCESSED" ] || [ -z "$REST" ]; then
    echo "Usage: ./merge_shuffle.sh processed.jsonl rest.jsonl [output]"
    exit 1
fi

TMP=$(mktemp)

# 合并
cat "$PROCESSED" "$REST" > "$TMP"

# 打乱
shuf "$TMP" > "$OUTPUT"

rm "$TMP"

echo "Merged and shuffled -> $OUTPUT"