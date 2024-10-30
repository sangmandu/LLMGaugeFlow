import ast
import json
import math
import os
import random
import re
import time
from collections import defaultdict

import jsonlines
import numpy as np
import pandas as pd
import shortuuid
from fastchat.llm_judge.common import (NEED_REF_CATS, check_data,
                                       load_judge_prompts, load_model_answers,
                                       load_questions, one_score_pattern,
                                       one_score_pattern_backup,
                                       play_a_match_single, temperature_config)
from fastchat.llm_judge.gen_judgment import (make_judge_single,
                                             make_match_single)
from mediapipe.tasks import python
from mediapipe.tasks.python import text

from llmgaugeflow import const
from llmgaugeflow.benchmarks.benchmark import Benchmark
from llmgaugeflow.llm_client import LLMClient
from settings import EVAL_RESULTS_DIR_PATH


class KoMTBench(Benchmark):
    def __init__(self, model):
        self.model = model

        # generate
        self.question_file = f"./llmgaugeflow/submodules/KoMT-Bench/FastChat/fastchat/llm_judge/data/mt_bench/question_ko.jsonl"
        self.question_begin = None
        self.question_end = None
        self.num_choices = 1
        self.model_kwargs = {
            "max_tokens": 1024,
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
        self.generate_file = os.path.join(
            EVAL_RESULTS_DIR_PATH, f"./KoMT-Bench/generated/{self.model}.jsonl"
        )

        # evaluate
        self.answer_dir = os.path.join(
            EVAL_RESULTS_DIR_PATH, "./KoMT-Bench/generated/*/"
        )
        self.ref_answer_dir = "./llmgaugeflow/submodules/KoMT-Bench/FastChat/fastchat/llm_judge/data/mt_bench/reference_answer_ko"
        self.judge_file = "./llmgaugeflow/submodules/KoMT-Bench/FastChat/fastchat/llm_judge/data/judge_prompts_ko.jsonl"
        self.judge_model = "gpt-4"  # "gpt-4-0613"
        self.parallel = 42
        self.output_file = os.path.join(
            EVAL_RESULTS_DIR_PATH, f"./KoMT-Bench/evaluated/{self.model}_single.jsonl"
        )

        # scoring
        self.detect_model = "./llmgaugeflow/benchmarks/KoMTBench/detector.tflite"
        self.final_file = os.path.join(
            EVAL_RESULTS_DIR_PATH,
            f"./KoMT-Bench/evaluated/{self.model}_single_final.jsonl",
        )
        self.score_dir = os.path.join(
            EVAL_RESULTS_DIR_PATH, f"./KoMT-Bench/score/{self.model}"
        )

    @classmethod
    def name(self):
        return const.Benchmark.KoMTBench.value

    @classmethod
    def evaluation_method(self):
        return const.EvaluationMethod.Judge.value

    def update_file_path(self, model_source):
        if const.ModelSource.OPENAI.is_same(model_source):
            self.generate_file = os.path.join(
                EVAL_RESULTS_DIR_PATH,
                f"./KoMT-Bench/generated/openai/{self.model}.jsonl",
            )
            self.output_file = os.path.join(
                EVAL_RESULTS_DIR_PATH,
                f"./KoMT-Bench/evaluated/openai/{self.model}_single.jsonl",
            )
            self.final_file = os.path.join(
                EVAL_RESULTS_DIR_PATH,
                f"./KoMT-Bench/evaluated/openai/{self.model}_single_final.jsonl",
            )
            self.score_dir = os.path.join(
                EVAL_RESULTS_DIR_PATH, f"./KoMT-Bench/score/openai/{self.model}"
            )
        elif const.ModelSource.ANTHROPIC.is_same(model_source):
            self.generate_file = os.path.join(
                EVAL_RESULTS_DIR_PATH,
                f"./KoMT-Bench/generated/anthropic/{self.model}.jsonl",
            )
            self.output_file = os.path.join(
                EVAL_RESULTS_DIR_PATH,
                f"./KoMT-Bench/evaluated/anthropic/{self.model}_single.jsonl",
            )
            self.final_file = os.path.join(
                EVAL_RESULTS_DIR_PATH,
                f"./KoMT-Bench/evaluated/anthropic/{self.model}_single_final.jsonl",
            )
            self.score_dir = os.path.join(
                EVAL_RESULTS_DIR_PATH, f"./KoMT-Bench/score/anthropic/{self.model}"
            )

    def generate(self, model_source, *args, **kwargs):
        self.update_file_path(model_source)

        llm = LLMClient(model_source, *args, **kwargs)
        if self.model not in llm.available_models:
            raise ValueError(f"Model {self.model} not found in {llm.available_models}")

        questions = load_questions(
            self.question_file, self.question_begin, self.question_end
        )
        random.shuffle(questions)

        PROMPT_STRATEGY = defaultdict(list)
        for question in questions:
            PROMPT_STRATEGY[question["category"]].append(question)

        answer_choices = defaultdict(list)
        for category, questions in PROMPT_STRATEGY.items():
            if category in temperature_config:
                self.model_kwargs["temperature"] = temperature_config[category]
                if self.model_kwargs["temperature"] < 1e-4:
                    self.model_kwargs["temperature"] = 0  # do_sample = False
            else:
                self.model_kwargs["temperature"] = 0.7

            for i in range(self.num_choices):
                self.model_kwargs["seed"] = i

                def format_single_turn_question(question):
                    qs = question["turns"][0]
                    return [{"role": "user", "content": qs}]

                single_turn_questions = list(
                    map(format_single_turn_question, questions)
                )
                single_turn_outputs = llm.run(
                    self.model, single_turn_questions, self.model_kwargs
                )

                def format_double_turn_question(idx):
                    single_turn_output = single_turn_outputs[idx]
                    if single_turn_output is None:
                        single_turn_output = "ERROR"
                    return [
                        {"role": "user", "content": questions[idx]["turns"][0]},
                        {"role": "assistant", "content": single_turn_output},
                        {"role": "user", "content": questions[idx]["turns"][1]},
                    ]

                multi_turn_questions = list(
                    map(format_double_turn_question, range(len(questions)))
                )
                multi_turn_outputs = llm.run(
                    self.model, multi_turn_questions, self.model_kwargs
                )

                for idx, question in enumerate(questions):
                    turns = [single_turn_outputs[idx], multi_turn_outputs[idx]]
                    answer_choices[question["question_id"]].append(
                        {"index": i, "turns": turns}
                    )

        if os.path.exists(self.generate_file):
            os.remove(self.generate_file)

        for question_id, choices in answer_choices.items():
            # Dump answers
            os.makedirs(os.path.dirname(self.generate_file), exist_ok=True)
            with open(
                os.path.expanduser(self.generate_file), "a", encoding="utf-8"
            ) as fout:
                ans_json = {
                    "question_id": question_id,
                    "answer_id": shortuuid.uuid(),
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")

    def evaluate(self, *args, **kwargs):
        llm = LLMClient("OpenAI", *args, **kwargs)
        if self.judge_model not in llm.available_models:
            raise ValueError(
                f"Model {self.judge_model} not found in {llm.available_models}"
            )

        # Load questions
        questions = load_questions(
            self.question_file, self.question_begin, self.question_end
        )

        # Load answers
        model_answers = load_model_answers(self.answer_dir)
        ref_answers = load_model_answers(self.ref_answer_dir)

        # Load judge
        judge_prompts = load_judge_prompts(self.judge_file)

        models = [self.model.split("/")[-1]]

        mode = "single"
        judges = make_judge_single(self.judge_model, judge_prompts)
        baseline_model = None

        check_data(questions, model_answers, ref_answers, models, judges)

        question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
        question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

        if os.path.exists(self.output_file):
            os.remove(self.output_file)

        # Make matches
        matches = []
        for matches in [
            make_match_single(
                question_default,
                models,
                model_answers,
                judges["default"],
                baseline_model,
            ),
            make_match_single(
                question_math,
                models,
                model_answers,
                judges["math"],
                baseline_model,
                ref_answers,
            ),
            make_match_single(
                question_default,
                models,
                model_answers,
                judges["default-mt"],
                baseline_model,
                multi_turn=True,
            ),
            make_match_single(
                question_math,
                models,
                model_answers,
                judges["math-mt"],
                baseline_model,
                ref_answers,
                multi_turn=True,
            ),
        ]:
            np.random.seed(0)
            np.random.shuffle(matches)

            # Play matches
            judge_messages = []
            for match in matches:
                question, answer, judge, ref_answer, multi_turn = (
                    match.question,
                    match.answer,
                    match.judge,
                    match.ref_answer,
                    match.multi_turn,
                )

                kwargs = {}
                model = judge.model_name
                if ref_answer is not None:
                    kwargs["ref_answer_1"] = ref_answer["choices"][0]["turns"][0]
                    if multi_turn:
                        kwargs["ref_answer_2"] = ref_answer["choices"][0]["turns"][1]

                if multi_turn:
                    user_prompt = judge.prompt_template["prompt_template"].format(
                        question_1=question["turns"][0],
                        question_2=question["turns"][1],
                        answer_1=answer["choices"][0]["turns"][0],
                        answer_2=answer["choices"][0]["turns"][1],
                        **kwargs,
                    )
                else:
                    user_prompt = judge.prompt_template["prompt_template"].format(
                        question=question["turns"][0],
                        answer=answer["choices"][0]["turns"][0],
                        **kwargs,
                    )

                system_prompt = judge.prompt_template["system_prompt"]
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": user_prompt},
                ]
                judge_messages.append(messages)

            judgments = llm.run(
                self.judge_model,
                judge_messages,
                {
                    "temperature": 0,
                    "max_tokens": 2048,
                },
            )

            for match, messages, judgment in zip(matches, judge_messages, judgments):
                question, model, answer, judge, ref_answer, multi_turn = (
                    match.question,
                    match.model,
                    match.answer,
                    match.judge,
                    match.ref_answer,
                    match.multi_turn,
                )
                user_prompt = messages[-1]["content"]

                score = -1
                if judge.prompt_template["output_format"] == "[[rating]]":
                    _match = re.search(one_score_pattern, judgment)
                    if not _match:
                        _match = re.search(one_score_pattern_backup, judgment)

                    if _match:
                        score = ast.literal_eval(_match.groups()[0])
                    else:
                        score = -1
                else:
                    raise ValueError(
                        f"invalid output format: {judge.prompt_template['output_format']}"
                    )

                question_id = question["question_id"]
                turn = 1 if not multi_turn else 2
                result = {
                    "question_id": question_id,
                    "model": model,
                    "judge": (judge.model_name, judge.prompt_template["name"]),
                    "user_prompt": user_prompt,
                    "judgment": judgment,
                    "score": score,
                    "turn": turn,
                    "tstamp": time.time(),
                }
                os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
                with open(self.output_file, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    def scoring(self):
        base_options = python.BaseOptions(model_asset_path=self.detect_model)
        options = text.LanguageDetectorOptions(base_options=base_options, max_results=2)
        detector = text.LanguageDetector.create_from_options(options)

        categorykeys = {}
        with jsonlines.open(self.question_file) as f:
            for line in f.iter():
                categorykeys[line["question_id"]] = line["category"]

        answerkeys = {}
        with jsonlines.open(self.generate_file) as f:
            for line in f.iter():
                answerkeys[line["question_id"]] = [
                    line["choices"][0]["turns"][0],
                    line["choices"][0]["turns"][1],
                ]

        with open(self.final_file, "w", encoding="utf-8") as e:
            with jsonlines.open(self.output_file) as f:
                for line in f.iter():
                    ori_socre = line["score"]
                    turn = line["turn"]
                    q_id = line["question_id"]
                    lang_result = detector.detect(answerkeys[q_id][turn - 1])
                    lang_result = lang_result.detections
                    line["language"] = {}

                    for detection in lang_result:
                        line["language"][
                            detection.language_code
                        ] = f"{detection.probability:.2f}"
                        lang_result = detection.language_code

                    # 영어로 답변 작성시 기존 score에 제곱근을 취합니다.
                    # 다만 거진 영어 100% 답변이 허용되는 json 형식의 답변인 138, 140 문항은 패널티 예외 조항으로 합니다
                    # 간혹가다 -1 점수가 나오는데 이도 제외합니다.
                    if lang_result != "ko" and q_id != 138 and q_id != 140:
                        try:
                            line["score"] = math.sqrt(ori_socre)
                        except:
                            pass
                    json.dump(line, e, ensure_ascii=False)
                    e.write("\n")

        file = pd.read_json(
            self.final_file, orient="records", encoding="utf-8-sig", lines=True
        )
        file["category"] = file["question_id"].apply(lambda x: categorykeys[x])
        file["turn"] = file["turn"].apply(lambda x: "Single" if x == 1 else "Multi")
        file = file[file["score"] != -1]

        ret = {
            "prompt_strategy": [],  # default, n-shot, cot
            "category": [],  # grammar, math, coding
            "turn": [],  # Single, Multi
            "score": [],
        }
        prompt_strategy = "default"
        total_single_scores = []
        total_multi_scores = []
        for category in file["category"].unique().tolist():
            for turn in file["turn"].unique().tolist():
                score = file[(file["category"] == category) & (file["turn"] == turn)][
                    "score"
                ].mean()
                ret["prompt_strategy"].append(prompt_strategy)
                ret["category"].append(category)
                ret["turn"].append(turn)
                ret["score"].append(score)

                if turn == "Single":
                    total_single_scores.append(score)
                else:
                    total_multi_scores.append(score)

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

        os.makedirs(self.score_dir, exist_ok=True)
        df = pd.DataFrame(ret)
        df.to_csv(os.path.join(self.score_dir, "score.csv"), index=False)
