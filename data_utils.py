import glob
import os

import numpy as np
import pandas as pd

from settings import EVAL_RESULTS_DIR_PATH

NO_EVAL_SCORE = "N/A"


def update_dataframe():
    dataframe = None
    for benchmark_name in os.listdir(EVAL_RESULTS_DIR_PATH):
        benchmark_dir_path = os.path.join(EVAL_RESULTS_DIR_PATH, benchmark_name)
        if os.path.isdir(benchmark_dir_path):
            score_paths = glob.glob(
                os.path.join(benchmark_dir_path, "score/*/*/score.csv")
            )
            for score_path in score_paths:
                model_name = score_path.split("score/")[-1].replace("/score.csv", "")

                _df = pd.read_csv(score_path)
                _df["benchmark"] = benchmark_name
                _df["model"] = model_name

                if dataframe is None:
                    dataframe = _df
                else:
                    dataframe = pd.concat([dataframe, _df])
    return dataframe


def get_model_list(dataframe):
    return (
        dataframe["model"].unique().tolist()
        if dataframe is not None and not dataframe.empty
        else []
    )


def get_benchmark_list(dataframe, sample=False):
    if dataframe is not None and not dataframe.empty:
        benchmark_list = dataframe["benchmark"].unique().tolist()
        return benchmark_list[0] if sample else benchmark_list
    return None if sample else []


def get_filtered_dataset(dataframe, models, benchmarks):
    if models is None:
        models = get_model_list(dataframe)

    if benchmarks is None:
        benchmarks = get_benchmark_list(dataframe)

    ret = {"Model": []}
    ret.update({bm: [] for bm in benchmarks})
    for model in models:
        ret["Model"].append(model)
        for benchmark in benchmarks:
            score = dataframe[
                (dataframe["model"] == model)
                & (dataframe["benchmark"] == benchmark)
                & (dataframe["category"] == "Overall")
                & (dataframe["turn"] == "Avg")
                & (dataframe["score"] != NO_EVAL_SCORE)
            ]["score"].values
            if len(score) == 0:
                ret[benchmark].append(NO_EVAL_SCORE)
            else:
                score = max(score)
                ret[benchmark].append(str(f"{score:.4f}"))
    return pd.DataFrame(ret)


def get_detail_benchmark_dataset(dataframe, models, benchmark):
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    if models is None:
        models = get_model_list(dataframe)

    if benchmark is None:
        benchmark = get_benchmark_list(dataframe, sample=True)

    df = dataframe[
        (dataframe["model"].isin(models))
        & (dataframe["benchmark"] == benchmark)
        & (dataframe["category"] != "Overall")
    ]
    if df.empty:
        return df

    na_model_list = sorted(set(models) - set(df["model"].unique().tolist()))
    if len(na_model_list) > 0:
        ret = {col: [] for col in df.columns}
        for prompt_strategy in df["prompt_strategy"].unique().tolist():
            for category in df["category"].unique().tolist():
                for turn in df["turn"].unique().tolist():
                    for model in na_model_list:
                        ret["prompt_strategy"].append(prompt_strategy)
                        ret["category"].append(category)
                        ret["turn"].append(turn)
                        ret["score"].append(NO_EVAL_SCORE)
                        ret["benchmark"].append(benchmark)
                        ret["model"].append(model)
        na_df = pd.DataFrame(ret)
        df = pd.concat([df, na_df]).reset_index(drop=True)
    return df


def sorted_by_overall(df):
    df["numeric_values"] = pd.to_numeric(df["Overall"], errors="coerce").fillna(-np.inf)
    df = df.sort_values(by="numeric_values", ascending=False)
    df = df.drop(columns=["numeric_values"])
    return df


def filter_best_scores_only(df):
    valid_scores = df[df["Overall"] != NO_EVAL_SCORE].copy()
    valid_scores["Overall"] = pd.to_numeric(valid_scores["Overall"], errors="coerce")

    best_scores_df = valid_scores.loc[valid_scores.groupby("Model")["Overall"].idxmax()]
    best_scores_df["Overall"] = best_scores_df["Overall"].apply(
        lambda x: str(f"{x:.4f}")
    )

    na_scores_df = df[
        (df["Overall"] == NO_EVAL_SCORE)
        & (~df["Model"].isin(best_scores_df["Model"].values))
    ]
    if not na_scores_df.empty:
        na_scores_df = na_scores_df.groupby("Model", as_index=False).first()

        best_scores_df = pd.concat([best_scores_df, na_scores_df]).reset_index(
            drop=True
        )
    return best_scores_df


def get_detail_row_data_and_column_defs(
    df, prompt_strategy="All", turn="All", toggle_show_all_models=False
):
    prompt_strategy_list = df["prompt_strategy"].unique().tolist()
    model_list = df["model"].unique().tolist()
    category_list = df["category"].unique().tolist()
    turn_list = df["turn"].unique().tolist()
    columns = ["Model"] + category_list + turn_list + ["Overall", "Prompt Strategy"]

    ret = {c: [] for c in columns}
    for _prompt_strategy in prompt_strategy_list:
        if prompt_strategy != "All" and prompt_strategy != _prompt_strategy:
            continue

        for model in model_list:
            sample = {"Prompt Strategy": _prompt_strategy, "Model": model}

            score = df[
                (df["prompt_strategy"] == _prompt_strategy)
                & (df["model"] == model)
                & (df["score"] != NO_EVAL_SCORE)
            ]["score"].values
            score = (
                str(f"{sum(score)/len(score):.4f}") if len(score) > 0 else NO_EVAL_SCORE
            )
            sample["Overall"] = score

            if not toggle_show_all_models and score == NO_EVAL_SCORE:
                continue

            for category in category_list:
                if turn == "All":
                    score = df[
                        (df["prompt_strategy"] == _prompt_strategy)
                        & (df["model"] == model)
                        & (df["category"] == category)
                        & (df["score"] != NO_EVAL_SCORE)
                    ]["score"].values
                else:
                    score = df[
                        (df["prompt_strategy"] == _prompt_strategy)
                        & (df["model"] == model)
                        & (df["category"] == category)
                        & (df["turn"] == turn)
                        & (df["score"] != NO_EVAL_SCORE)
                    ]["score"].values

                score = (
                    str(f"{sum(score)/len(score):.4f}")
                    if len(score) > 0
                    else NO_EVAL_SCORE
                )
                sample[category] = score

            for _turn in turn_list:
                score = df[
                    (df["prompt_strategy"] == _prompt_strategy)
                    & (df["model"] == model)
                    & (df["turn"] == _turn)
                    & (df["score"] != NO_EVAL_SCORE)
                ]["score"].values

                score = (
                    str(f"{sum(score)/len(score):.4f}")
                    if len(score) > 0
                    else NO_EVAL_SCORE
                )
                sample[_turn] = score

            for col, value in sample.items():
                ret[col].append(value)

    detail_df = pd.DataFrame(ret)
    detail_df = filter_best_scores_only(detail_df)
    detail_df = sorted_by_overall(detail_df)

    columnDefs = [
        (
            {"field": col, "headerName": col, "pinned": "left"}
            if col == "Model"
            else {"field": col, "headerName": col}
        )
        for col in detail_df.columns
    ]
    return detail_df.to_dict("records"), columnDefs
