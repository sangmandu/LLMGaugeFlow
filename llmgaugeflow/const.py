from enum import Enum


class ModelSource(Enum):
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    OPENSOURCE = "OpenSource"

    @classmethod
    def is_valid_source(cls, source: str) -> bool:
        return source not in cls._value2member_map_

    def is_same(self, source: str) -> bool:
        return source and self.value.lower() == source.lower()

    @classmethod
    def to_list(cls):
        return [member.value for member in cls]


class Benchmark(Enum):
    LogicKor = "LogicKor"
    KoMTBench = "KoMT-Bench"

    @classmethod
    def is_valid_source(cls, source: str) -> bool:
        return source not in cls._value2member_map_

    @classmethod
    def to_list(cls):
        return [member.value for member in cls]

    def is_same(self, source: str) -> bool:
        return source and self.value.lower() == source.lower()

    @classmethod
    def get_class(cls, model_source):
        if cls.is_valid_source(model_source):
            raise ValueError(f"Invalid model source: {model_source}")
        elif cls.LogicKor.is_same(model_source):
            from llmgaugeflow.benchmarks import LogicKor

            return LogicKor
        elif cls.KoMTBench.is_same(model_source):
            from llmgaugeflow.benchmarks import KoMTBench

            return KoMTBench
        else:
            raise ValueError(f"No class found for {model_source}")


class EvaluationMethod(Enum):
    Judge = "judge"

    @classmethod
    def to_list(cls):
        return [member.value for member in cls]

    def is_same(self, source: str) -> bool:
        return source and self.value.lower() == source.lower()


class EvaluationTaskStatus(Enum):
    Pending = "Pending"
    Started = "Started"
    Generating = "Generating"
    Evaluating = "Evaluating"
    Scoring = "Scoring"
    Completed = "Completed"
    Failed = "Failed"

    def is_same(self, source: str) -> bool:
        return source and self.value.lower() == source.lower()
