#!/bin/bash

# Launch one worker per split file.
for i in {0..11}
do
   echo "Starting worker $i for part_$i.jsonl ..."
   # Use batch size 128 to reduce GPU memory pressure.
   # Replace the script path if a different extractor is used.
   python gen_np.py \
     --input-jsonl part_$i.jsonl \
     --output-jsonl output_part_$i.jsonl & 
done

# Wait for all background workers to finish.
wait
echo "All processing tasks are complete!"
