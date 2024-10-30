from abc import abstractmethod

CONCURRENT_REQUESTS = 10  # You can adjust this to a number that works best for you
CHECKPOINT_INTERVAL = 50  # Save after every n generations
MAX_RETRIES = 1
BACKOFF_FACTOR = 30  # The factor by which the delay increases during each retry


class LLMClient:
    @property
    @abstractmethod
    def client(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def aclient(self):
        raise NotImplementedError

    @abstractmethod
    def generate(self, model, messages, model_kwargs={}):
        raise NotImplementedError

    @abstractmethod
    async def agenerate(self, model, messages, model_kwargs={}, attempt=1):
        raise NotImplementedError
