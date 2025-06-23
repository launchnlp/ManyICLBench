export HF_HOME=YOUR_HF_HOME
export MODEL_PATH=Llama-3.1-8B
export GPUS=4
export MAX_MODEL_LENGTH=131072
export HF_TOKEN=YOUR_HF_TOKEN
python serve_vllm.py \
    --model=${MODEL_PATH} \
    --tensor-parallel-size=${GPUS} \
    --max-model-len ${MAX_MODEL_LENGTH} \
    --port 8080 \
    --enable-prefix-caching \
    --trust-remote-code \
    --disable-log-requests \
    --disable-log-stats \
    --disable-custom-all-reduce