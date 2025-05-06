#!/bin/bash

MODEL_PATH="/home/omnisky/userfile_2/wangfj/TinyLLaVA/checkpoints/llava_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-20250421"
MODEL_NAME="tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-20250421"
EVAL_DIR="/home/omnisky/userfile_2/wangfengjuan/TinyLLaVABench-main/eval"

python -m tinyllava.eval.model_vqa_mmmu \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/mmmu/anns_for_eval.json \
    --image-folder $EVAL_DIR/mmmu/all_images \
    --answers-file $EVAL_DIR/mmmu/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python scripts/convert_answer_to_mmmu.py \
    --answers-file $EVAL_DIR/mmmu/answers/$MODEL_NAME.jsonl \
    --answers-output $EVAL_DIR/mmmu/answers/"$MODEL_NAME"_output.json

cd $EVAL_DIR/mmmu/eval

python main_eval_only.py --output_path $EVAL_DIR/mmmu/answers/"$MODEL_NAME"_output.json
