import os

EVAL_RESULTS_DIR_PATH = "./eval_results"
os.makedirs(EVAL_RESULTS_DIR_PATH, exist_ok=True)
EVAL_TASKS_DIR_PATH = "./evaluation_tasks"
os.makedirs(EVAL_TASKS_DIR_PATH, exist_ok=True)
EVAL_TASK_PREFIX = "auto_eval_task_"
