MODEL_PATH="/home/omnisky/userfile_2/wangfj/TinyLLaVA/checkpoints/llava_factory/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-20250421"
MODEL_NAME="tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune-20250421"
EVAL_DIR="/home/omnisky/userfile_2/wangfengjuan/TinyLLaVABench-main/eval"


python -m tinyllava.eval.model_vqa_vizwiz \
    --model-path  $MODEL_PATH \
    --question-file $EVAL_DIR/vizwiz/llava_test.jsonl \
    --image-folder $EVAL_DIR/vizwiz/test \
    --answers-file $EVAL_DIR/vizwiz/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode phi

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file $EVAL_DIR/vizwiz/llava_test.jsonl \
    --result-file $EVAL_DIR/vizwiz/answers/$MODEL_NAME.jsonl \
    --result-upload-file $EVAL_DIR/vizwiz/answers_upload/$MODEL_NAME.json