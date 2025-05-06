#!/bin/bash

MODEL_PATH="/home/omnisky/userfile_2/wangfj/TinyLLaVA/checkpoints/llava_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-20250421"
MODEL_NAME="tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-20250421"
EVAL_DIR="/home/omnisky/userfile_2/wangfengjuan/TinyLLaVABench-main/eval"

python -m tinyllava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/mm-vet/llava-mm-vet.jsonl \
    --image-folder $EVAL_DIR/mm-vet/images \
    --answers-file $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

mkdir -p $EVAL_DIR/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src $EVAL_DIR/mm-vet/answers/$MODEL_NAME.jsonl \
    --dst $EVAL_DIR/mm-vet/results/$MODEL_NAME.json
