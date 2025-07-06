#!/bin/bash

FINETUNED_PATH="/root/workspace/finetuned_sd"
OUTPUT_DIR="/root/workspace/result"
SAVE_DIR="/root/shared-storage/result"

mkdir $OUTPUT_DIR

IMAGE_COUNT=1000
BATCH_SIZE=4
HEIGHT=512
WIDTH=512
STEPS=50
SEED=42

echo "We are ready."

python sample.py --model_path $FINETUNED_PATH \
--output_dir $OUTPUT_DIR --gpu_id 0 \
--label_start 1 --label_end 15 \
--image_count_per_label $IMAGE_COUNT --batch_size $BATCH_SIZE \
--height $HEIGHT --width $WIDTH \
--steps $STEPS --seed $SEED &

python sample.py --model_path $FINETUNED_PATH \
--output_dir $OUTPUT_DIR --gpu_id 1 \
--label_start 16 --label_end 30 \
--image_count_per_label $IMAGE_COUNT --batch_size $BATCH_SIZE \
--height $HEIGHT --width $WIDTH \
--steps $STEPS --seed $SEED &

python sample.py --model_path $FINETUNED_PATH \
--output_dir $OUTPUT_DIR --gpu_id 2 \
--label_start 31 --label_end 45 \
--image_count_per_label $IMAGE_COUNT --batch_size $BATCH_SIZE \
--height $HEIGHT --width $WIDTH \
--steps $STEPS --seed $SEED &

python sample.py --model_path $FINETUNED_PATH \
--output_dir $OUTPUT_DIR --gpu_id 3 \
--label_start 46 --label_end 60 \
--image_count_per_label $IMAGE_COUNT --batch_size $BATCH_SIZE \
--height $HEIGHT --width $WIDTH \
--steps $STEPS --seed $SEED &

echo "Every subprocess start..."

wait

echo "Every subprocess over..."

cp -r $OUTPUT_DIR $SAVE_DIR

echo "copy over!!!"