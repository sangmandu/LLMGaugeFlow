import asyncio
import os

import openai
from openai import AsyncOpenAI, OpenAI

from llmgaugeflow.llm_cache import llm_cache
from llmgaugeflow.llm_client.client import (BACKOFF_FACTOR, MAX_RETRIES,
                                            LLMClient)


class OpenAIClient(LLMClient):
    def __init__(self, *args, **kwargs):
        self.is_openai = True

        openai_api_key = kwargs.get("api_key", "EMPTY")
        openai_api_base = kwargs.get("api_base_url", None)
        if openai_api_key == "EMPTY" and openai_api_base is None:
            raise ValueError("OpenAI API key or base URL is required")
        elif openai_api_base is not None:
            self.is_openai = False
            openai_api_base = os.path.join(openai_api_base.rstrip("/"), "v1")

        self._client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self._aclient = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            max_retries=MAX_RETRIES,
        )

        models = self._client.models.list()
        self.available_models = sorted([model.id for model in models.data])

    @property
    def client(self):
        return self._client

    @property
    def aclient(self):
        return self._aclient

    @llm_cache
    def generate(self, model, messages, model_kwargs={}):
        try:
            if model not in self.available_models:
                raise ValueError(f"Model {model} not available")
            elif self.is_openai:
                model_kwargs = {
                    "temperature": model_kwargs.get("temperature", 0),
                    "max_tokens": model_kwargs.get("max_tokens", 1024),
                }

            response = self._client.chat.completions.create(
                messages=messages, model=model, stream=False, **model_kwargs
            )
            generated_text = response.choices[0].message.content
            return generated_text
        except TimeoutError:
            print(f"Request timed out for prompt: {messages}")
            return None
        except openai.RateLimitError:
            print(f"Rate limit exceeded. Retrying in 30 ~ 60 seconds...")
            return None
        except openai.APITimeoutError:
            print(f"Request timed out for prompt: {messages}")
            return None
        except openai.APIConnectionError as e:
            print(f"API connection error for prompt: {messages}")
            return None
        except openai.PermissionDeniedError as e:
            print(f"PermissionDeniedError occurred: {e}")
            print(type(e))
            print(messages)
            return None
        except openai.OpenAIError as e:
            print(f"OpenAIError occurred: {e}")
            print(type(e))
            print(messages)
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            print(type(e))
            print(messages)
            return None

    @llm_cache
    async def agenerate(self, model, messages, model_kwargs={}, attempt=1):
        try:
            if model not in self.available_models:
                raise ValueError(f"Model {model} not available")
            elif self.is_openai:
                model_kwargs = {
                    "temperature": model_kwargs.get("temperature", 0),
                    "max_tokens": model_kwargs.get("max_tokens", 1024),
                }

            response = await self._aclient.chat.completions.create(
                messages=messages, model=model, stream=False, **model_kwargs
            )
            generated_text = response.choices[0].message.content.strip()
            return generated_text
        except asyncio.exceptions.TimeoutError:
            print(f"Request timed out for prompt: {messages}")
            return None
        except TimeoutError:
            print(f"Request timed out for prompt: {messages}")
            return None
        except openai.RateLimitError:
            if attempt <= MAX_RETRIES:
                sleep_time = BACKOFF_FACTOR**attempt
                print(f"Rate limit exceeded. Retrying in {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)
                return await self.agenerate(model, messages, model_kwargs, attempt + 1)
            else:
                print(f"Failed to translate after {MAX_RETRIES} retries: {messages}")
                return None
        except openai.APITimeoutError:
            print(f"Request timed out for prompt: {messages}")
            return None
        except openai.APIConnectionError as e:
            print(f"API connection error for prompt: {messages}")
            return None
        except openai.PermissionDeniedError as e:
            print(f"PermissionDeniedError occurred: {e}")
            print(type(e))
            print(messages)
            return None
        except openai.OpenAIError as e:
            print(f"OpenAIError occurred: {e}")
            print(type(e))
            print(messages)
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            print(type(e))
            print(messages)
            return None
