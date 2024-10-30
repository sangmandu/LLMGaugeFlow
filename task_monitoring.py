import json
import os
import threading
import time
from queue import Queue

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from llmgaugeflow.const import Benchmark, EvaluationTaskStatus, ModelSource
from settings import EVAL_TASK_PREFIX, EVAL_TASKS_DIR_PATH


def get_evaluation_tasks():
    task_data_ret = []
    for filename in os.listdir(EVAL_TASKS_DIR_PATH):
        if filename.startswith(EVAL_TASK_PREFIX) and filename.endswith(".json"):
            task_file = os.path.join(EVAL_TASKS_DIR_PATH, filename)
            with open(task_file, "r") as f:
                task_data = json.load(f)
                task_data["task_file"] = task_file
            task_data_ret.append(task_data)
    return task_data_ret


class EvaluationTask:
    def __init__(self, task_file, task_data):
        self.task_file = task_file
        self.task_data = task_data

    def update_task_status(self, status):
        self.task_data["task_status"] = status
        with open(self.task_file, "w") as f:
            f.write(json.dumps(self.task_data))

    def run_evaluation(self):
        self.update_task_status(EvaluationTaskStatus.Started.value)

        benchmark = self.task_data.get("benchmark")
        model = self.task_data.get("model")
        model_source = self.task_data.get("model_source")
        llm_client_kwargs = self.task_data.get("llm_client_kwargs", {})
        if ModelSource.OPENAI.is_same(model_source):
            openai_api_key = llm_client_kwargs.get("api_key")
        else:
            openai_api_key = self.task_data.get("openai_api_key")

        try:
            benchmark_cls = Benchmark.get_class(benchmark)
            benchmark_instance = benchmark_cls(model=model)

            self.update_task_status(EvaluationTaskStatus.Generating.value)
            benchmark_instance.generate(model_source=model_source, **llm_client_kwargs)

            self.update_task_status(EvaluationTaskStatus.Evaluating.value)
            benchmark_instance.evaluate(api_key=openai_api_key)

            self.update_task_status(EvaluationTaskStatus.Scoring.value)
            benchmark_instance.scoring()

            self.update_task_status(EvaluationTaskStatus.Completed.value)
        except Exception as e:
            print(e)
            self.update_task_status(EvaluationTaskStatus.Failed.value)
            raise e


TASK_QUEUE = Queue()
for task_data in get_evaluation_tasks():
    task_file = task_data.get("task_file")
    task_status = task_data.get("task_status")
    if task_file:
        if not any(
            [
                EvaluationTaskStatus.Completed.is_same(task_status),
                EvaluationTaskStatus.Failed.is_same(task_status),
            ]
        ):
            task_data["task_status"] = EvaluationTaskStatus.Pending.value
            with open(task_file, "w") as f:
                f.write(json.dumps(task_data))
            TASK_QUEUE.put(task_file)


def process_queue():
    while True:
        try:
            task_file = TASK_QUEUE.get()
            check_and_run_evaluation(task_file)
            TASK_QUEUE.task_done()
        except:
            continue


def check_and_run_evaluation(task_file):
    with open(task_file, "r") as f:
        task_data = json.load(f)

    task_status = task_data.get("task_status")
    if EvaluationTaskStatus.Pending.is_same(task_status):
        evaluation_task = EvaluationTask(task_file, task_data)
        evaluation_task.run_evaluation()


class TaskHandler(FileSystemEventHandler):
    def on_created(self, event):
        if self.is_eval_task_file(event.src_path):
            TASK_QUEUE.put(event.src_path)

    def is_eval_task_file(self, file_path):
        filename = os.path.basename(file_path)
        return filename.startswith(EVAL_TASK_PREFIX) and filename.endswith(".json")


def start_monitoring():
    event_handler = TaskHandler()
    observer = Observer()
    observer.schedule(event_handler, path=EVAL_TASKS_DIR_PATH, recursive=False)
    observer.start()

    try:
        threading.Thread(target=process_queue, daemon=True).start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
