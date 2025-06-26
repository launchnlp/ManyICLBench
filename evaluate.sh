export TASK=ARC-Challenge
export MAX_CONTEXT_LENGTH=128000
export MODE=full
export MODEL_FRAMEWORK=vllm
export MODEL_PATH=Llama-3.1-8B-Instruct
export BATCH_SIZE=1
export HOST=localhost
export PORT=8080
TASKS=("banking77"
    "goEmotions"
    "dialogRE"
    "trec_50"
    "clinc150"
    "MATH-algebra"
    "MATH-geometry"
    "MATH-counting_and_probability"
    "MATH-number_theory"
    "BBH-geometric_shapes"
    "BBH-salient_translation_error_detection"
    "BBH-word_sorting"
    "BBH-dyck_languages"
    "GPQA"
    "GPQA_cot"
    "ARC-Easy"
    "ARC-Challenge"
    "GSM8K"
    "XLSUM"
    "MT_Kurdish"
    "MT_Chinese"
    "MT_Spanish"
)
for TASK in "${TASKS[@]}"
do
    echo "Evaluating task: $TASK"
    python call_api.py \
        --dataset ${TASK} \
        --context_length ${MAX_CONTEXT_LENGTH} \
        --mode ${MODE} \
        --server_type ${MODEL_FRAMEWORK} \
        --model_name_or_path ${MODEL_PATH} \
        --batch_size ${BATCH_SIZE} \
        --server_host $HOST \
        --server_port $PORT \
        --seeds 3
done
