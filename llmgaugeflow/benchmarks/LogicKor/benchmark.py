import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from llmgaugeflow import const
from llmgaugeflow.benchmarks.benchmark import Benchmark
from llmgaugeflow.llm_client import LLMClient
from llmgaugeflow.submodules.LogicKor.evaluator import is_hidden, process_file
from llmgaugeflow.submodules.LogicKor.templates import PROMPT_STRATEGY
from settings import EVAL_RESULTS_DIR_PATH


@dataclass
class Args:
    model_output_dir: str


class LogicKor(Benchmark):
    def __init__(self, model):
        self.model = model

        # generate
        self.input_path = "./llmgaugeflow/submodules/LogicKor/questions.jsonl"
        self.model_kwargs = {
            "temperature": 0,
            "max_tokens": 1024,
            # "skip_special_tokens": True,
            "stop": [
                "<|endoftext|>",
                "[INST]",
                "[/INST]",
                "<|im_end|>",
                "<|end|>",
                "<|eot_id|>",
                "<end_of_turn>",
                "<eos>",
            ],
        }
        self.generated_dir = os.path.join(
            EVAL_RESULTS_DIR_PATH, f"./LogicKor/generated/{self.model}"
        )

        # evaluate
        self.judge_model = "gpt-4-1106-preview"
        self.evaluated_dir = os.path.join(
            EVAL_RESULTS_DIR_PATH, f"./LogicKor/evaluated/{self.model}"
        )
        self.threads = 42

        # scoring
        self.score_dir = os.path.join(
            EVAL_RESULTS_DIR_PATH, f"./LogicKor/score/{self.model}"
        )

    @classmethod
    def name(self):
        return const.Benchmark.LogicKor.value

    @classmethod
    def evaluation_method(self):
        return const.EvaluationMethod.Judge.value

    def update_file_path(self, model_source):
        if const.ModelSource.OPENAI.is_same(model_source):
            self.generated_dir = os.path.join(
                EVAL_RESULTS_DIR_PATH, f"./LogicKor/generated/openai/{self.model}"
            )
            self.evaluated_dir = os.path.join(
                EVAL_RESULTS_DIR_PATH, f"./LogicKor/evaluated/openai/{self.model}"
            )
            self.score_dir = os.path.join(
                EVAL_RESULTS_DIR_PATH, f"./LogicKor/score/openai/{self.model}"
            )
        elif const.ModelSource.ANTHROPIC.is_same(model_source):
            self.generated_dir = os.path.join(
                EVAL_RESULTS_DIR_PATH, f"./LogicKor/generated/anthropic/{self.model}"
            )
            self.evaluated_dir = os.path.join(
                EVAL_RESULTS_DIR_PATH, f"./LogicKor/evaluated/anthropic/{self.model}"
            )
            self.score_dir = os.path.join(
                EVAL_RESULTS_DIR_PATH, f"./LogicKor/score/anthropic/{self.model}"
            )

    def generate(self, model_source, *args, **kwargs):
        self.update_file_path(model_source)

        llm = LLMClient(model_source, *args, **kwargs)
        if self.model not in llm.available_models:
            raise ValueError(f"Model {self.model} not found in {llm.available_models}")

        df_questions = pd.read_json(
            self.input_path, orient="records", encoding="utf-8-sig", lines=True
        )
        if not os.path.exists(self.generated_dir):
            os.makedirs(self.generated_dir)

        for strategy_name, prompts in PROMPT_STRATEGY.items():

            def format_single_turn_question(question):
                return prompts + [{"role": "user", "content": question[0]}]

            single_turn_questions = df_questions["questions"].map(
                format_single_turn_question
            )
            single_turn_outputs = llm.run(
                self.model, single_turn_questions, self.model_kwargs
            )

            def format_double_turn_question(question, single_turn_output):
                return prompts + [
                    {"role": "user", "content": question[0]},
                    {"role": "assistant", "content": single_turn_output},
                    {"role": "user", "content": question[1]},
                ]

            multi_turn_questions = df_questions[["questions", "id"]].apply(
                lambda x: format_double_turn_question(
                    x["questions"], single_turn_outputs[x["id"] - 1]
                ),
                axis=1,
            )
            multi_turn_outputs = llm.run(
                self.model, multi_turn_questions, self.model_kwargs
            )

            df_output = pd.DataFrame(
                {
                    "id": df_questions["id"],
                    "category": df_questions["category"],
                    "questions": df_questions["questions"],
                    "outputs": list(zip(single_turn_outputs, multi_turn_outputs)),
                    "references": df_questions["references"],
                }
            )
            df_output.to_json(
                os.path.join(self.generated_dir, f"{strategy_name}.jsonl"),
                orient="records",
                lines=True,
                force_ascii=False,
            )

    def evaluate(self, *args, **kwargs):
        llm = LLMClient("OpenAI", *args, **kwargs)
        if self.judge_model not in llm.available_models:
            raise ValueError(
                f"Model {self.judge_model} not found in {llm.available_models}"
            )

        input_dir = Path(self.generated_dir)
        output_dir = Path(self.evaluated_dir)

        # Filter out hidden files
        json_files = [
            file for file in input_dir.rglob("*.jsonl") if not is_hidden(file)
        ]
        print(f"Found {len(json_files)} JSON files to process")

        args = Args(model_output_dir=self.generated_dir)

        for file_path in json_files:
            output_file_path = output_dir / file_path.relative_to(input_dir)
            if output_file_path.exists():
                print(f"이미 평가 완료.. : {file_path}")
                continue

            process_file(
                llm.client, file_path, output_dir, self.judge_model, self.threads, args
            )
            time.sleep(20)  # to handle ratelimit!

    def scoring(self):
        input_dir = Path(self.evaluated_dir)
        output_dir = Path(self.score_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filter out hidden files
        json_files = [
            file for file in input_dir.rglob("*.jsonl") if not is_hidden(file)
        ]
        print(f"Found {len(json_files)} JSON files to process")

        ret = {
            "prompt_strategy": [],
            "category": [],
            "turn": [],
            "score": [],
        }
        for file_path in json_files:
            print(f"\n\nProcessing {file_path}")

            prompt_strategy = file_path.stem
            print(f"Prompt strategy: {prompt_strategy}")

            file = pd.read_json(
                file_path, orient="records", encoding="utf-8-sig", lines=True
            )

            category_scores = {}
            for item in file.to_dict(orient="records"):
                category = item["category"]
                single_score = item["query_single"]["judge_score"]
                multi_score = item["query_multi"]["judge_score"]

                if category not in category_scores:
                    category_scores[category] = {
                        "single_scores": [],
                        "multi_scores": [],
                    }

                category_scores[category]["single_scores"].append(single_score)
                category_scores[category]["multi_scores"].append(multi_score)

            total_single_scores = []
            total_multi_scores = []
            for category, scores in category_scores.items():
                avg_single = sum(scores["single_scores"]) / len(scores["single_scores"])
                avg_multi = sum(scores["multi_scores"]) / len(scores["multi_scores"])

                total_single_scores.extend(scores["single_scores"])
                total_multi_scores.extend(scores["multi_scores"])

                ret["prompt_strategy"].append(prompt_strategy)
                ret["category"].append(category)
                ret["turn"].append("Single")
                ret["score"].append(avg_single)

                ret["prompt_strategy"].append(prompt_strategy)
                ret["category"].append(category)
                ret["turn"].append("Multi")
                ret["score"].append(avg_multi)

            if len(total_single_scores) == 0:
                avg_total_single = 0
            else:
                avg_total_single = sum(total_single_scores) / len(total_single_scores)

            if len(total_multi_scores) == 0:
                avg_total_multi = 0
            else:
                avg_total_multi = sum(total_multi_scores) / len(total_multi_scores)

            avg_total = (avg_total_single + avg_total_multi) / 2

            ret["prompt_strategy"].append(prompt_strategy)
            ret["category"].append("Overall")
            ret["turn"].append("Single")
            ret["score"].append(avg_total_single)

            ret["prompt_strategy"].append(prompt_strategy)
            ret["category"].append("Overall")
            ret["turn"].append("Multi")
            ret["score"].append(avg_total_multi)

            ret["prompt_strategy"].append(prompt_strategy)
            ret["category"].append("Overall")
            ret["turn"].append("Avg")
            ret["score"].append(avg_total)

        df = pd.DataFrame(ret)
        df.to_csv(os.path.join(self.score_dir, "score.csv"), index=False)
