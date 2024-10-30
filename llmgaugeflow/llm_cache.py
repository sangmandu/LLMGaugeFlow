import hashlib
import os
from functools import wraps
from typing import List

import aiofiles

CACHE_DIR = "./.cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def simple_hash(text):
    if text is None:
        return None

    return hashlib.md5(text.encode("utf-8")).hexdigest()


def llm_cache(func):
    @wraps(func)
    async def wrapper(
        cls, model: str, messages: List[dict], model_kwargs={}, *args, **kwargs
    ):
        hash = simple_hash(str(messages) + str(model_kwargs))
        body = "_".join(["__".join(model.split("/")), hash])
        cache_path = os.path.join(CACHE_DIR, body + ".cache")
        try:
            if os.path.exists(cache_path):
                print(f"Hit cache path: {cache_path}")
                async with aiofiles.open(cache_path) as fp:
                    return await fp.read()
        except Exception as e:
            print(f"Failed to read cache file. path: {cache_path}, error: {e}")

        generated_text = await func(cls, model, messages, model_kwargs, *args, **kwargs)
        if generated_text:
            async with aiofiles.open(cache_path, "wt") as fp:
                await fp.write(generated_text)
            print(f"Created cache file. path: {cache_path}")
        return generated_text

    return wrapper
