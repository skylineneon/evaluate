import os
# os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
import argparse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
import math
import json
from fastapi.responses import StreamingResponse

TIMEOUT_KEEP_ALIVE = 5


class Model_Server:
    def __init__(self, model_path: str) -> None:
        parser = argparse.ArgumentParser()
        parser = AsyncEngineArgs.add_cli_args(parser)
        args = parser.parse_args()

        args.model = model_path
        args.trust_remote_code = True
        self.engine_args = AsyncEngineArgs.from_cli_args(args)
        self.engine_args.max_model_len = 12288  # 32768
        self.engine_args.max_num_seqs = 10
        self.engine_args.enforce_eager = True
        self.engine_args.gpu_memory_utilization = 0.9
        self.engine_args.tensor_parallel_size = 1
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        self.tokenizer = self.engine.engine.tokenizer

    def get_tokenizer(self):
        return self.tokenizer

    async def model_infer(self, request):
        json_dic = await request.json()

        # 设定参数值
        mark = json_dic['mark'] if json_dic['mark'] else ""
        n = json_dic['n'] if json_dic['n'] else 1
        best_of = json_dic['best_of'] if json_dic['best_of'] else None
        presence_penalty = json_dic['presence_penalty'] if json_dic['presence_penalty'] else 0.0
        frequency_penalty = json_dic['frequency_penalty'] if json_dic['frequency_penalty'] else 0.0
        repetition_penalty = json_dic['repetition_penalty'] if json_dic['repetition_penalty'] else 1.0
        temperature = json_dic['temperature'] if json_dic['temperature'] else 0
        top_p = json_dic['top_p'] if json_dic['top_p'] else 1.0
        top_k = json_dic['top_k'] if json_dic['top_k'] else -1
        min_p = json_dic['min_p'] if json_dic['min_p'] else 0.0
        use_beam_search = json_dic['use_beam_search'] if json_dic['use_beam_search'] else False
        length_penalty = json_dic['length_penalty'] if json_dic['length_penalty'] else 1.0
        early_stopping = json_dic['early_stopping'] if json_dic['early_stopping'] else False
        stop = json_dic['stop'] if json_dic['stop'] else []
        stop_token_ids = json_dic['stop_token_ids'] if json_dic['stop_token_ids'] else []
        include_stop_str_in_output = json_dic['include_stop_str_in_output'] if json_dic[
            'include_stop_str_in_output'] else False
        ignore_eos = json_dic['ignore_eos'] if json_dic['ignore_eos'] else False
        max_tokens = json_dic['max_new_tokens'] if json_dic['max_new_tokens'] else 16
        logprobs = json_dic['logprobs'] if json_dic['logprobs'] else None
        prompt_logprobs = json_dic['prompt_logprobs'] if json_dic['prompt_logprobs'] else None
        skip_special_tokens = json_dic['skip_special_tokens'] if json_dic['skip_special_tokens'] else True
        spaces_between_special_tokens = json_dic['spaces_between_special_tokens'] if json_dic[
            'spaces_between_special_tokens'] else True
        logits_processors = json_dic['logits_processors'] if json_dic['logits_processors'] else None

        model_ins = ''
        if json_dic['messages']:
            for msg in json_dic['messages']:
                if msg['role'] == 'user':
                    model_ins += '<|im_start|>user\n' + msg['content'] + '<|im_end|>\n'
                elif msg['role'] == 'system':
                    model_ins += '<|im_start|>system\n' + msg['content'] + '<|im_end|>\n'
                else:
                    model_ins += '<|im_start|>assistant\n' + msg['content'] + '<|im_end|>\n'
            model_ins += '<|im_start|>assistant\n'
        else:
            model_ins = json_dic['model_ins']
        
        print("len(model_ins): ", len(model_ins))
        if len(model_ins) > 11288:
            return json.dumps({"error": "the content length should be < 11288"})

        sampling_params = SamplingParams(
            n=n,
            best_of=best_of,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            use_beam_search=use_beam_search,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            stop=stop,
            stop_token_ids=stop_token_ids,
            include_stop_str_in_output=include_stop_str_in_output,
            ignore_eos=ignore_eos,
            max_tokens=max_tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            logits_processors=logits_processors,
            seed=100
        )

        
        if json_dic['stream']:
            return StreamingResponse(
                self.get_generate_stream(request, model_ins, sampling_params, mark),
            )
        else:
            async for response in self.get_generate_stream(request, model_ins, sampling_params, mark):
                continue
            return json.loads(response)

    async def get_generate_stream(self, request, model_ins, sampling_params, mark):
        request_id = random_uuid()
        results_generator = self.engine.generate(
            inputs=model_ins,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        counter = 0
        async for request_output in results_generator:
            if await request.is_disconnected():
                await self.engine.abort(request_id)  # 删除占存
                print('context.is_disconnected()中断,返回空文本')
                yield '{}\n'
                return

            if math.isnan(request_output.outputs[0].cumulative_logprob):
                await self.engine.abort(request_id)
                yield json.dumps({"result": '400_content_filter'}, ensure_ascii=False)
                return

            text_outputs = [i.text for i in request_output.outputs]
            counter += 1
            if counter % 10 == 0:
                yield json.dumps({
                    "mark": mark,
                    "result": text_outputs
                }, ensure_ascii=False) + "\n"

        yield json.dumps({
            "mark": mark,
            "result": text_outputs
        }, ensure_ascii=False) + "\n"

