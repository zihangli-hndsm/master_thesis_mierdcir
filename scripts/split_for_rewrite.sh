#!/bin/bash

INPUT=$1
SEED=${2:-114514}

if [ -z "$INPUT" ]; then
    echo "Usage: ./split_jsonl.sh big.jsonl [seed]"
    exit 1
fi

# 输出文件
SAMPLE="sample_10p.jsonl"
REST="rest_90p.jsonl"

# 清空旧文件
> "$SAMPLE"
> "$REST"

# 随机抽样
awk -v seed="$SEED" '
BEGIN {
    srand(seed)
}
{
    if (rand() < 0.1)
        print > "'"$SAMPLE"'"
    else
        print > "'"$REST"'"
}
' "$INPUT"

echo "Done."
echo "10% sample -> $SAMPLE"
echo "90% rest   -> $REST"