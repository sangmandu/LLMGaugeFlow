from abc import ABC, abstractmethod


class Benchmark(ABC):
    @classmethod
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def evaluation_method(self):
        raise NotImplementedError

    @abstractmethod
    def generate(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

    @abstractmethod
    def scoring(self):
        raise NotImplementedError
