#!/bin/bash

MODEL_PATH="/home/omnisky/userfile_2/wangfj/TinyLLaVA/checkpoints/llava_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-20250421"
MODEL_NAME="tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-20250421"
EVAL_DIR="/home/omnisky/userfile_2/wangfengjuan/TinyLLaVABench-main/eval"

python -m tinyllava.eval.model_vqa_pope \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --image-folder $EVAL_DIR/pope/val2014 \
    --answers-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python tinyllava/eval/eval_pope.py \
    --annotation-dir $EVAL_DIR/pope/coco \
    --question-file $EVAL_DIR/pope/llava_pope_test.jsonl \
    --result-file $EVAL_DIR/pope/answers/$MODEL_NAME.jsonl
