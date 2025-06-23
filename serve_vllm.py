# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from https://github.com/vllm-project/vllm/blob/v0.4.0/vllm/entrypoints/api_server.py

import argparse
import json
import os
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from huggingface_hub import hf_hub_download

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None
old_model_name = None

model_mapping = {
    'Llama-3.1-8B': 'meta-llama/Meta-Llama-3.1-8B',
    'gemma-2-9b': 'google/gemma-2-9b',
    'Mistral-Nemo-Base': 'mistralai/Mistral-Nemo-Base-2407',
    'glm-4-9b-chat': 'THUDM/glm-4-9b-chat',
    'glm-4-9b': 'THUDM/glm-4-9b',
    'Llama-3.1-8B-Instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'Llama-3.1-70B-Instruct': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'Mistral-Nemo-Instruct': 'mistralai/Mistral-Nemo-Instruct-2407',
    'Mistral-Large-Instruct': 'mistralai/Mistral-Large-Instruct-2407',
    'Mistral-Large-Instruct-GGUF': "/scratch/wangluxy_owned_root/wangluxy_owned1/zkjzou/cache/huggingface/Mistral-Large-Instruct-2407-Q8_0",
    'gemma-2-9b-it': 'google/gemma-2-9b-it',
    'command-r':'CohereForAI/c4ai-command-r-v01',
    'Meta-Llama-3.1-405B-Instruct-4bit': 'hugging-quants/Meta-Llama-3.1-405B-Instruct-GPTQ-INT4',
    'Qwen2-7B-Instruct':'Qwen/Qwen2-7B-Instruct',
    'Qwen2-72B-Instruct':'Qwen/Qwen2-72B-Instruct',
    'Qwen2-7B-Instruct-AWQ':'Qwen/Qwen2-7B-Instruct-AWQ',
    'Qwen2-72B-Instruct-AWQ':'Qwen/Qwen2-72B-Instruct-AWQ',
    'Llama-3.1-70B-Instruct-AWQ':"hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    'Llama-3.1-8B-Instruct-AWQ':"hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    "Mistral-Large-Instruct-AWQ": "casperhansen/mistral-large-instruct-2407-awq",
    "Mistral-Nemo-Instruct-AWQ": "casperhansen/mistral-nemo-instruct-2407-awq",
    "command-r-4bit":"CohereForAI/c4ai-command-r-v01-4bit",
    "command-r-plus-4bit": "CohereForAI/c4ai-command-r-plus-4bit",
    'Jamba-1.5-Mini': 'ai21labs/AI21-Jamba-1.5-Mini',
    'Phi-3-Mini-Instruct': 'microsoft/Phi-3-mini-128k-instruct',
    'Phi-3-Small-Instruct': 'microsoft/Phi-3-small-128k-instruct',
    'Phi-3-Medium-Instruct': 'microsoft/Phi-3-medium-128k-instruct'
}

@app.get("/health")
async def health() -> Response:
    """Health check."""
    job_id = os.getenv('SLURM_JOB_ID')
    return Response(content=job_id, status_code=200)


@app.put("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = engine.generate(prompt,
                                        sampling_params,
                                        request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output
    assert final_output is not None
    text_outputs = [output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    old_model_name = engine_args.model
    engine_args.model = model_mapping[engine_args.model]
    engine_args.enable_prefix_caching = True
    if engine_args.model == 'bartowski/Mistral-Large-Instruct-2407-GGUF':
        repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
        filename = "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
        model = hf_hub_download(repo_id, filename=filename)
    if 'awq' in engine_args.model.lower():
        engine_args.quantization = 'awq_marlin'
    if 'Qwen' in engine_args.model and engine_args.max_model_len > 32000:
        engine_args.rope_scaling = {
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "rope_type": "yarn"
        }
    if 'Jamba' in engine_args.model:
        os.environ['VLLM_FUSED_MOE_CHUNK_SIZE']='32768' 
        engine_args.quantization = 'experts_int8'
        engine_args.gpu_memory_utilization = 0.85
        engine_args.enable_prefix_caching = False
    if 'Phi' in engine_args.model:
        engine_args.enable_prefix_caching = False
        engine_args.enable_chunked_prefill = False
        if 'medium' in engine_args.model:
            engine_args.quantization = 'fp8'
    engine_args.tokenizer = engine_args.model
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)