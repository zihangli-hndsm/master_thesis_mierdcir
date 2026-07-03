#!/bin/bash

PROCESSED=$1
REST=$2
OUTPUT=${3:-merged_shuffled.jsonl}

if [ -z "$PROCESSED" ] || [ -z "$REST" ]; then
    echo "Usage: ./merge_shuffle.sh processed.jsonl rest.jsonl [output]"
    exit 1
fi

TMP=$(mktemp)

# Merge all split outputs into one file.
cat "$PROCESSED" "$REST" > "$TMP"

# Shuffle merged records to remove split-order bias.
shuf "$TMP" > "$OUTPUT"

rm "$TMP"

echo "Merged and shuffled -> $OUTPUT"