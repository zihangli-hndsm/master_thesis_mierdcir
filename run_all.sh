#!/bin/bash

# 循环启动 12 个进程
for i in {0..11}
do
   echo "正在启动进程 $i 处理 part_$i.jsonl ..."
   # 使用 --batch-size 128 避免显存溢出
   # 这里的 python 后面跟的是你原本的提取脚本
   python gen_np.py \
     --input-jsonl part_$i.jsonl \
     --output-jsonl output_part_$i.jsonl & 
done

# 等待所有后台任务完成
wait
echo "所有处理任务已完成！"
