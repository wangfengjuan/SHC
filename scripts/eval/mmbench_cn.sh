SPLIT="mmbench_dev_cn_20231003"

MODEL_PATH="/home/omnisky/userfile_2/wangfj/TinyLLaVA/checkpoints/llava_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-20250421"
MODEL_NAME="tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-20250421"
EVAL_DIR="/home/omnisky/userfile_2/wangfengjuan/TinyLLaVABench-main/eval"

python -m tinyllava.eval.model_vqa_mmbench \
    --model-path $MODEL_PATH \
    --question-file $EVAL_DIR/mmbench_ch/$SPLIT.tsv \
    --answers-file $EVAL_DIR/mmbench_ch/answers/$SPLIT/$MODEL_NAME.jsonl \
    --lang cn \
    --single-pred-prompt \
    --temperature 0 \
    

mkdir -p $EVAL_DIR/mmbench_ch/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file $EVAL_DIR/mmbench_ch/$SPLIT.tsv \
    --result-dir $EVAL_DIR/mmbench_ch/answers/$SPLIT \
    --upload-dir $EVAL_DIR/mmbench_ch/answers_upload/$SPLIT \
    --experiment $MODEL_NAME
