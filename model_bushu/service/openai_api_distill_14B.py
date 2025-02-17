import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import asyncio
import copy
import json
from contextlib import asynccontextmanager
import importlib
import inspect
from prometheus_client import make_asgi_app
import fastapi
import uvicorn
from http import HTTPStatus
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse
from vllm.logger import init_logger
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import LoRAModulePath, BaseModelPath
import vllm
from vllm.config import ModelConfig

TIMEOUT_KEEP_ALIVE = 5

openai_serving_chat: OpenAIServingChat = None
openai_serving_completion: OpenAIServingCompletion = None
logger = init_logger(__name__)


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    async def _force_log():
        while True:
            await asyncio.sleep(10)
            await engine.do_log_stats()

    if not engine_args.disable_log_stats:
        asyncio.create_task(_force_log())

    yield


app = fastapi.FastAPI(lifespan=lifespan)


class LoRAParserAction(argparse.Action):

    def __call__(self, namespace, values):
        lora_list = []
        for item in values:
            name, path = item.split('=')
            lora_list.append(LoRAModulePath(name=name, local_path=path))
        setattr(namespace, self.dest, lora_list)


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
    parser.add_argument("--host", type=str, default='0.0.0.0', help="host name")
    parser.add_argument("--port", type=int, default=8510, help="port number")
    parser.add_argument(
        "--uvicorn-log-level",
        type=str,
        default="info",
        choices=['debug', 'info', 'warning', 'error', 'critical', 'trace'],
        help="log level for uvicorn")
    parser.add_argument("--allow-credentials",
                        action="store_true",
                        help="allow credentials")
    parser.add_argument("--allowed-origins",
                        type=json.loads,
                        default=["*"],
                        help="allowed origins")
    parser.add_argument("--allowed-methods",
                        type=json.loads,
                        default=["*"],
                        help="allowed methods")
    parser.add_argument("--allowed-headers",
                        type=json.loads,
                        default=["*"],
                        help="allowed headers")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=
        "If provided, the server will require this key to be presented in the header."
    )
    parser.add_argument(
        "--lora-modules",
        type=str,
        default=None,
        nargs='+',
        action=LoRAParserAction,
        help=
        "LoRA module configurations in the format name=path. Multiple modules can be specified."
    )
    parser.add_argument("--chat-template",
                        type=str,
                        default=None,
                        help="The file path to the chat template, "
                             "or the template in single-line form "
                             "for the specified model")
    parser.add_argument("--response-role",
                        type=str,
                        default="assistant",
                        help="The role name to return if "
                             "`request.add_generation_prompt=true`.")
    parser.add_argument("--ssl-keyfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL key file")
    parser.add_argument("--ssl-certfile",
                        type=str,
                        default=None,
                        help="The file path to the SSL cert file")
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument(
        "--middleware",
        type=str,
        action="append",
        default=[],
        help="Additional ASGI middleware to apply to the app. "
             "We accept multiple --middleware arguments. "
             "The value should be an import path. "
             "If a function is provided, vLLM will add it to the server using @app.middleware('http'). "
             "If a class is provided, vLLM will add it to the server using app.add_middleware(). "
    )
    parser = AsyncEngineArgs.add_cli_args(parser)
    return parser.parse_args()


metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc):
    err = openai_serving_chat.create_error_response(message=str(exc))
    return JSONResponse(err.model_dump(), status_code=HTTPStatus.BAD_REQUEST)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request):
    print(request)
    request_copy = copy.deepcopy(request)
    request_copy = request_copy.json()
    generator = await openai_serving_chat.create_chat_completion(
        request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(),
                            status_code=generator.code)
    if request.stream:
        try:
            logger.info(f'openai_request:{request_copy}')
            logger.info('*' * 100)
        except Exception as e:
            logger.error('openai_stream_error:', str(e))
        return StreamingResponse(content=generator,
                                 media_type="text/event-stream")
    else:
        logger.info(f'openai_request:{request_copy}')
        logger.info(f'openai_response:{str(generator.model_dump())}')
        logger.info('*' * 100)
        return JSONResponse(content=generator.model_dump())


if __name__ == "__main__":
    args = parse_args()
    args.model = '/DATA/LLM_model/deepseek/DeepSeek-R1-Distill-Qwen-14B'
    args.served_model_name = 'DeepSeek-R1-Distill-Qwen-14B'
    args.gpu_memory_utilization = 0.6
    args.tensor_parallel_size = 1

    args.base_model_path = BaseModelPath(name=args.served_model_name, model_path=args.model)
    args.max_model_len = 12288
    args.stream = False
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    if token := os.environ.get("VLLM_API_KEY") or args.api_key:

        @app.middleware("http")
        async def authentication(request: Request, call_next):
            if not request.url.path.startswith("/v1"):
                return await call_next(request)
            if request.headers.get("Authorization") != "Bearer " + token:
                return JSONResponse(content={"error": "Unauthorized"},
                                    status_code=401)
            return await call_next(request)

    for middleware in args.middleware:
        module_path, object_name = middleware.rsplit(".", 1)
        imported = getattr(importlib.import_module(module_path), object_name)
        if inspect.isclass(imported):
            app.add_middleware(imported)
        elif inspect.iscoroutinefunction(imported):
            app.middleware("http")(imported)
        else:
            raise ValueError(
                f"Invalid middleware {middleware}. Must be a function or a class."
            )

    logger.info(f"vLLM API server version {vllm.__version__}")
    logger.info(f"args: {args}")

    if args.served_model_name is not None:
        served_model = args.served_model_name
    else:
        served_model = args.model
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    model_config = ModelConfig(model=args.model,
                               task="auto",
                               tokenizer=args.model,
                               tokenizer_mode='auto',
                               trust_remote_code=True,
                               dtype='auto',
                               seed=100)
    openai_serving_chat = OpenAIServingChat(engine_client=engine,
                                            model_config=model_config,
                                            base_model_paths=[args.base_model_path],
                                            response_role=args.response_role,
                                            lora_modules=args.lora_modules,
                                            prompt_adapters=None,
                                            request_logger=None,
                                            chat_template=args.chat_template)

    openai_serving_completion = OpenAIServingCompletion(
        engine_client=engine,
        model_config=model_config,
        base_model_paths=[args.base_model_path],
        lora_modules=args.lora_modules,
        prompt_adapters=None,
        request_logger=None)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.uvicorn_log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
