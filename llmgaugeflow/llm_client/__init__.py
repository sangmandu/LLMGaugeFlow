import asyncio
import time
from typing import List

from llmgaugeflow.const import ModelSource
from llmgaugeflow.llm_client.client import (CHECKPOINT_INTERVAL,
                                            CONCURRENT_REQUESTS)
from llmgaugeflow.llm_client.openai import OpenAIClient


class LLMClient:
    def __init__(self, model_source, *args, **kwargs):
        if ModelSource.is_valid_source(model_source):
            raise ValueError(f"Invalid model source: {model_source}")
        elif ModelSource.ANTHROPIC.is_same(model_source):
            raise NotImplementedError("Anthropic model is not supported yet")
        else:
            self._client = OpenAIClient(*args, **kwargs)

    @property
    def client(self):
        return self._client.client

    @property
    def aclient(self):
        return self._client.aclient

    @property
    def available_models(self):
        return self._client.available_models

    def generate(self, model: str, messages: List[dict], model_kwargs: dict = {}):
        return self._client.generate(model, messages, model_kwargs)

    async def agenerate(
        self,
        model: str,
        messages: List[dict],
        model_kwargs: dict = {},
        attempt: int = 1,
    ):
        return await self._client.agenerate(model, messages, model_kwargs, attempt)

    async def arun(
        self, model: str, message_list: List[tuple], model_kwargs: dict = {}
    ):
        tasks = [
            asyncio.create_task(self.agenerate(model, message, model_kwargs))
            for _, message in message_list
        ]
        async with asyncio.Semaphore(CONCURRENT_REQUESTS):
            candidate_answers = await asyncio.gather(*tasks)
        return candidate_answers

    def run(
        self,
        model: str,
        message_list: List[list],
        model_kwargs: dict = {},
        max_try_num: int = 2,
    ):
        message_list = [(idx, messages) for idx, messages in enumerate(message_list)]

        results = {}
        for str_idx in range(0, len(message_list), CHECKPOINT_INTERVAL):
            end_idx = str_idx + CHECKPOINT_INTERVAL
            _message_list = message_list[str_idx:end_idx]

            try_count = 0
            while try_count < max_try_num:
                try_count += 1
                candidate_answers = asyncio.run(
                    self.arun(model, _message_list, model_kwargs)
                )

                # if generated_answers value is NaN, send it back to prompt_list
                failed_prompt = []
                for _message, _candidate_answer in zip(
                    _message_list, candidate_answers
                ):
                    if _candidate_answer is None:
                        failed_prompt.append(_message)
                    else:
                        _idx = _message[0]
                        results[_idx] = _candidate_answer
                print(
                    f"Result Total: {len(results)}/{len(message_list)} / Failed: {len(failed_prompt)}/{len(_message_list)}"
                )

                if try_count < max_try_num and len(failed_prompt) > 0:
                    _message_list = failed_prompt
                    print(f"Retry {len(_message_list)} failed generation...")
                    time.sleep(1)
                elif len(failed_prompt) == 0:
                    break

            if len(failed_prompt) != 0:
                for fail in failed_prompt:
                    results[fail[0]] = None
                print(f"Failed to generate {len(failed_prompt)} prompts.")

        print(f"generated {len(results)} results")
        generated_results = [
            res for _, res in sorted(results.items(), key=lambda x: x[0])
        ]
        if None in generated_results:
            raise ValueError("Failed to generate some prompts. Please check the logs.")
        return generated_results
