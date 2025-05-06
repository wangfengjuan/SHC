#!/bin/bash


MODEL_PATH="/home/omnisky/userfile_2/wangfj/TinyLLaVA/checkpoints/llava_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-1121"
MODEL_NAME="tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-1121"
EVAL_DIR="/home/omnisky/userfile_2/wangfengjuan/TinyLLaVABench-main/eval"


python -m tinyllava.eval.model_vqa \
    --model-path $MODEL_PATH \
    --question-file c/llava-bench-in-the-wild/questions.jsonl \
    --image-folder $EVAL_DIR/llava-bench-in-the-wild/images \
    --answers-file $EVAL_DIR/llava-bench-in-the-wild/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

mkdir -p $EVAL_DIR/llava-bench-in-the-wild/reviews

python tinyllava/eval/eval_gpt_review_bench.py \
    --question $EVAL_DIR/llava-bench-in-the-wild/questions.jsonl \
    --context $EVAL_DIR/llava-bench-in-the-wild/context.jsonl \
    --rule tinyllava/eval/llava-bench-in-the-wild/table/rule.json \
    --answer-list \
        $EVAL_DIR/llava-bench-in-the-wild/answers_gpt4.jsonl \
        $EVAL_DIR/llava-bench-in-the-wild/answers/llava-v1.5-13b.jsonl \
    --output \
        $EVAL_DIR/llava-bench-in-the-wild/reviews/llava-v1.5-13b.jsonl

python llava/eval/summarize_gpt_review.py -f $EVAL_DIR/llava-bench-in-the-wild/reviews/llava-v1.5-13b.jsonl


