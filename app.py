import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(__file__), "llmgaugeflow", "submodules", "LogicKor")
)
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "llmgaugeflow",
        "submodules",
        "KoMT-Bench",
        "FastChat",
    )
)

import json
import threading
import time
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import shortuuid
from dash import Dash, Input, Output, State, callback_context, dcc, html

from data_utils import (NO_EVAL_SCORE, get_benchmark_list,
                        get_detail_benchmark_dataset,
                        get_detail_row_data_and_column_defs,
                        get_filtered_dataset, get_model_list, update_dataframe)
from layout.botton import download_button_style, refresh_button_style
from llmgaugeflow.const import (Benchmark, EvaluationMethod,
                                EvaluationTaskStatus, ModelSource)
from llmgaugeflow.llm_client import LLMClient
from settings import EVAL_TASK_PREFIX, EVAL_TASKS_DIR_PATH
from task_monitoring import get_evaluation_tasks, start_monitoring

TITLE = "All-in-One Benchmark"

app = Dash(
    __name__,
    title=TITLE,
    update_title=None,
    use_pages=True,
    suppress_callback_exceptions=True,
    prevent_initial_callbacks="initial_duplicate",
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

NAVBAR = dbc.Navbar(
    children=[
        dbc.Container(
            [
                dbc.NavbarBrand(
                    [
                        html.Img(src="assets/symbol_color.png", height="30px"),
                        " ",
                        html.Div(
                            TITLE,
                            className="d-none d-md-block",
                            style={
                                "color": "black",
                                "fontWeight": "bold",
                                "marginLeft": "10px",
                            },
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center"},
                ),
                dbc.Nav(
                    [
                        dbc.NavLink("🏆 Overall", href="/"),
                        dbc.NavLink("🔍 Detail", href="/detail"),
                        dbc.NavLink("📝 Evaluation", href="/evaluation"),
                        dbc.NavLink("🎯 Tracking", href="/tracking"),
                    ]
                ),
            ],
            fluid=True,
        )
    ],
    sticky="top",
    expand="lg",
    color="white",  # 배경색 흰색
    dark=False,  # Navbar 텍스트 어두운 색으로 설정 (dark=False)
)

app.layout = html.Div(
    [
        NAVBAR,  # Navbar 삽입
        dbc.Container(
            dash.page_container,
            fluid=True,
            style={"color": "white", "backgroundColor": "black"},
            className="p-3",
        ),
    ],
    style={"backgroundColor": "black"},  # 전체 페이지 배경 검정색으로 설정
)


################ Overall Callback #################


@app.callback(
    Output("overall-store-selected", "data"),
    Input("overall-refresh-button", "n_clicks"),
    State("overall-store-selected", "data"),
)
def update_overall_dataframe(refresh_n_clicks, data):
    dataframe = update_dataframe()
    if dataframe is None:
        return data

    dataframe = dataframe.to_dict("records")
    data["dataframe"] = dataframe
    return data


@app.callback(
    Output("overall-model-benchmark-collapse", "is_open"),
    [Input("overall-model-benchmark-header-btn", "n_clicks")],
    [State("overall-model-benchmark-collapse", "is_open")],
)
def overall_toggle_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@app.callback(
    [
        Output("overall-model", "options"),
        Output("overall-benchmark", "options"),
    ],
    Input("overall-refresh-button", "n_clicks"),
    State("overall-store-selected", "data"),
)
def update_overall_options(refresh_n_clicks, data):
    dataframe = pd.DataFrame(data.get("dataframe", []))

    model_list = get_model_list(dataframe)
    benchmark_list = get_benchmark_list(dataframe)

    return (
        [{"label": model, "value": model} for model in model_list],
        [{"label": benchmark, "value": benchmark} for benchmark in benchmark_list],
    )


@app.callback(
    [
        Output("overall-grid", "rowData"),
        Output("overall-grid", "columnDefs"),
        Output("overall-model", "value"),
        Output("overall-benchmark", "value"),
        Output("overall-store-selected", "data", allow_duplicate=True),
    ],
    [
        Input("overall-refresh-button", "n_clicks"),
        Input("overall-model", "value"),
        Input("overall-benchmark", "value"),
    ],
    State("overall-store-selected", "data"),
)
def update_overall_grid_data(
    refresh_n_clicks, selected_models, selected_benchmarks, data
):
    dataframe = pd.DataFrame(data.get("dataframe", []))

    if selected_models is None:
        selected_models = data.get("selected_models", get_model_list(dataframe))
    if selected_benchmarks is None:
        selected_benchmarks = data.get(
            "selected_benchmarks", get_benchmark_list(dataframe)
        )

    data["selected_models"] = selected_models
    data["selected_benchmarks"] = selected_benchmarks

    filtered_df = get_filtered_dataset(dataframe, selected_models, selected_benchmarks)
    column_defs = [{"field": "Model", "headerName": "Model", "pinned": "left"}] + [
        {"field": bm, "headerName": bm} for bm in selected_benchmarks
    ]
    return (
        filtered_df.to_dict("records"),
        column_defs,
        selected_models,
        selected_benchmarks,
        data,
    )


@app.callback(
    Output("overall-download-dataframe", "data"),
    Input("overall-download-button", "n_clicks"),
    State("overall-grid", "rowData"),
)
def download_overall_grid_data(download_n_clicks, row_data):
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "overall-download-button":
        dataframe = pd.DataFrame(row_data)
        if not dataframe.empty:
            return dcc.send_data_frame(dataframe.to_csv, "data.csv")


@app.callback(
    Output("overall-download-button", "style"),
    Output("overall-refresh-button", "style"),
    Input("overall-download-button", "n_clicks"),
    Input("overall-refresh-button", "n_clicks"),
    prevent_initial_call=True,
)
def update_overall_button_styles(download_clicks, refresh_clicks):
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    download_style = download_button_style.copy()
    refresh_style = refresh_button_style.copy()
    if triggered_id == "overall-download-button":
        download_style.update(
            {
                "background-color": "#086A87",
                "color": "slategray",
                "box-shadow": "none",
            }
        )
    elif triggered_id == "overall-refresh-button":
        refresh_style.update(
            {
                "background-color": "#0B6121",
                "color": "slategray",
                "box-shadow": "none",
            }
        )
    return download_style, refresh_style


@app.callback(
    Output("overall-download-button", "style", allow_duplicate=True),
    Output("overall-refresh-button", "style", allow_duplicate=True),
    Input("overall-download-button", "n_clicks"),
    Input("overall-refresh-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_overall_button_styles(download_clicks, refresh_clicks):
    download_style = download_button_style.copy()
    refresh_style = refresh_button_style.copy()
    time.sleep(0.2)
    return download_style, refresh_style


################ Detail Callback #################


@app.callback(
    Output("detail-store-selected", "data"),
    Input("detail-refresh-button", "n_clicks"),
    State("detail-store-selected", "data"),
)
def update_detail_dataframe(refresh_n_clicks, data):
    dataframe = update_dataframe()
    if dataframe is None:
        return data

    dataframe = dataframe.to_dict("records")
    data["dataframe"] = dataframe
    return data


@app.callback(
    Output("detail-model-benchmark-collapse", "is_open"),
    [Input("detail-model-benchmark-header-btn", "n_clicks")],
    [State("detail-model-benchmark-collapse", "is_open")],
)
def detail_toggle_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


@app.callback(
    [
        Output("detail-model", "options"),
        Output("detail-benchmark", "options"),
    ],
    Input("detail-refresh-button", "n_clicks"),
    State("detail-store-selected", "data"),
)
def update_detail_options(refresh_n_clicks, data):
    dataframe = pd.DataFrame(data.get("dataframe", []))

    model_list = get_model_list(dataframe)
    benchmark_list = get_benchmark_list(dataframe)

    return (
        [{"label": model, "value": model} for model in model_list],
        [{"label": benchmark, "value": benchmark} for benchmark in benchmark_list],
    )


@app.callback(
    [
        Output("detail-grid", "rowData"),
        Output("detail-grid", "columnDefs"),
        Output("detail-model", "value"),
        Output("detail-benchmark", "value"),
        Output("detail-prompt-strategy", "value"),
        Output("detail-turn", "value"),
        Output("detail-benchmark-title", "children"),
        Output("detail-prompt-strategy", "options"),
        Output("detail-turn", "options"),
        Output("detail-store-selected", "data", allow_duplicate=True),
    ],
    [
        Input("detail-model", "value"),
        Input("detail-benchmark", "value"),
        Input("detail-prompt-strategy", "value"),
        Input("detail-turn", "value"),
        Input("detail-show-all-models", "value"),
    ],
    State("detail-store-selected", "data"),
)
def update_detail_benchmark_options(
    selected_models,
    selected_benchmark,
    selected_prompt_strategy,
    selected_turn,
    toggle_show_all_models,
    data,
):
    dataframe = pd.DataFrame(data.get("dataframe", []))

    if selected_models is None:
        selected_models = data.get("selected_models", get_model_list(dataframe))
    if selected_benchmark is None:
        selected_benchmark = data.get(
            "selected_benchmark", get_benchmark_list(dataframe, sample=True)
        )

    data["selected_models"] = selected_models
    data["selected_benchmark"] = selected_benchmark

    benchmark_dataframe = get_detail_benchmark_dataset(
        dataframe, selected_models, selected_benchmark
    )

    prompt_strategy_options = []
    turn_options = []
    row_data = []
    column_defs = []
    if not benchmark_dataframe.empty:
        prompt_strategy_list = ["All"] + benchmark_dataframe[
            "prompt_strategy"
        ].unique().tolist()
        prompt_strategy_options = [
            {"label": strategy, "value": strategy} for strategy in prompt_strategy_list
        ]
        turn_list = ["All"] + benchmark_dataframe["turn"].unique().tolist()
        turn_options = [{"label": turn, "value": turn} for turn in turn_list]

        def get_valid_selection(value, options, default="All"):
            return value if value in options else default

        selected_prompt_strategy = get_valid_selection(
            selected_prompt_strategy or data.get("selected_prompt_strategy"),
            prompt_strategy_list,
        )
        selected_turn = get_valid_selection(
            selected_turn or data.get("selected_turn"),
            turn_list,
        )

        data["selected_prompt_strategy"] = selected_prompt_strategy
        data["selected_turn"] = selected_turn

        row_data, column_defs = get_detail_row_data_and_column_defs(
            benchmark_dataframe,
            selected_prompt_strategy,
            selected_turn,
            toggle_show_all_models[0] if toggle_show_all_models else False,
        )
    return (
        row_data,
        column_defs,
        selected_models,
        selected_benchmark,
        selected_prompt_strategy,
        selected_turn,
        selected_benchmark,
        prompt_strategy_options,
        turn_options,
        data,
    )


@app.callback(
    [
        Output("detail-graph", "figure"),
        Output("detail-graph", "style"),
    ],
    Input("detail-grid", "rowData"),
)
def update_detail_prompt_strategy_and_turn(row_data):
    benchmark_df = pd.DataFrame(row_data)
    if benchmark_df.empty:
        return {}, {"display": "none"}

    benchmark_df = benchmark_df[benchmark_df["Overall"] != NO_EVAL_SCORE]
    benchmark_df = benchmark_df.drop(
        columns=["Single", "Multi", "Overall", "Prompt Strategy"]
    )

    ret = {
        "model": [],
        "category": [],
        "score": [],
    }
    for _, row in benchmark_df.iterrows():
        model = row.pop("Model")
        for category, score in row.items():
            ret["model"].append(model)
            ret["category"].append(category)
            ret["score"].append(float(score))
    df_score = pd.DataFrame(ret)

    fig = px.line_polar(
        df_score,
        r="score",
        theta="category",
        line_close=True,
        color="model",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        range_r=[0, 10],
    )
    return fig, {"display": "block"}


@app.callback(
    Output("detail-download-dataframe", "data"),
    Input("detail-download-button", "n_clicks"),
    State("detail-grid", "rowData"),
)
def download_detail_grid_data(download_n_clicks, row_data):
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "detail-download-button":
        dataframe = pd.DataFrame(row_data)
        if not dataframe.empty:
            return dcc.send_data_frame(dataframe.to_csv, "data.csv")


@app.callback(
    Output("detail-download-button", "style"),
    Output("detail-refresh-button", "style"),
    Input("detail-download-button", "n_clicks"),
    Input("detail-refresh-button", "n_clicks"),
    prevent_initial_call=True,
)
def update_detail_button_styles(download_clicks, refresh_clicks):
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    download_style = download_button_style.copy()
    refresh_style = refresh_button_style.copy()
    if triggered_id == "detail-download-button":
        download_style.update(
            {
                "background-color": "#086A87",
                "color": "slategray",
                "box-shadow": "none",
            }
        )
    elif triggered_id == "detail-refresh-button":
        refresh_style.update(
            {
                "background-color": "#0B6121",
                "color": "slategray",
                "box-shadow": "none",
            }
        )
    return download_style, refresh_style


@app.callback(
    Output("detail-download-button", "style", allow_duplicate=True),
    Output("detail-refresh-button", "style", allow_duplicate=True),
    Input("detail-download-button", "n_clicks"),
    Input("detail-refresh-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_detail_button_styles(download_clicks, refresh_clicks):
    download_style = download_button_style.copy()
    refresh_style = refresh_button_style.copy()
    time.sleep(0.5)
    return download_style, refresh_style


################ Evaluation Callback #################


@app.callback(
    [
        Output("auto-eval-api-key-input", "style"),
        Output("auto-eval-endpoint-url-input", "style"),
    ],
    Input("auto-eval-model-source-dropdown", "value"),
)
def show_input_fields(model_source):
    if ModelSource.OPENAI.is_same(model_source):
        return (
            {"margin-bottom": "10px", "display": "block"},
            {"margin-bottom": "10px", "display": "none"},
        )
    elif ModelSource.OPENSOURCE.is_same(model_source):
        return (
            {"margin-bottom": "10px", "display": "none"},
            {"margin-bottom": "10px", "display": "block"},
        )
    else:
        return (
            {"margin-bottom": "10px", "display": "none"},
            {"margin-bottom": "10px", "display": "none"},
        )


@app.callback(
    [
        Output("auto-eval-model-dropdown", "options"),
        Output("auto-eval-model-dropdown", "style"),
    ],
    [
        Input("auto-eval-api-key-input", "value"),
        Input("auto-eval-endpoint-url-input", "value"),
    ],
    State("auto-eval-model-source-dropdown", "value"),
)
def show_model_fields(api_key, endpoint_url, model_source):
    if not model_source:
        return (
            [],
            {
                "margin-bottom": "10px",
                "color": "black",
                "display": "none",
            },
        )

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if endpoint_url:
        kwargs["api_base_url"] = endpoint_url

    if not kwargs:
        return (
            [],
            {
                "margin-bottom": "10px",
                "color": "black",
                "display": "none",
            },
        )

    try:
        client = LLMClient(model_source, **kwargs)
        return (
            [{"label": model, "value": model} for model in client.available_models],
            {
                "margin-bottom": "10px",
                "color": "black",
                "display": "block",
            },
        )
    except Exception as e:
        return (
            [],
            {
                "margin-bottom": "10px",
                "color": "black",
                "display": "none",
            },
        )


@app.callback(
    Output("auto-eval-benchmarks-dropdown", "style"),
    Input("auto-eval-model-dropdown", "value"),
)
def show_benchmark_fields(model):
    if model:
        return {"margin-bottom": "10px", "display": "block"}
    return {"margin-bottom": "10px", "display": "none"}


@app.callback(
    [
        Output("auto-eval-openai-api-key-input", "style"),
        Output("auto-eval-evaluate-button", "style"),
    ],
    Input("auto-eval-benchmarks-dropdown", "value"),
    State("auto-eval-model-source-dropdown", "value"),
)
def show_benchmark_fields(benchmarks, model_source):
    if benchmarks:
        if not ModelSource.OPENAI.is_same(model_source) and any(
            [
                EvaluationMethod.Judge.is_same(
                    Benchmark.get_class(benchmark).evaluation_method()
                )
                for benchmark in benchmarks
            ]
        ):
            return (
                {"margin-bottom": "10px", "display": "block"},
                {"margin-bottom": "10px", "display": "block", "white-space": "nowrap"},
            )
        return (
            {"margin-bottom": "10px", "display": "none"},
            {"margin-bottom": "10px", "display": "block", "white-space": "nowrap"},
        )
    else:
        return (
            {"margin-bottom": "10px", "display": "none"},
            {"margin-bottom": "10px", "display": "none", "white-space": "nowrap"},
        )


@app.callback(
    [
        Output("auto-eval-evaluation-output", "children"),
        Output("auto-eval-evaluate-button", "style", allow_duplicate=True),
    ],
    Input("auto-eval-evaluate-button", "n_clicks"),
    [
        State("auto-eval-model-source-dropdown", "value"),
        State("auto-eval-api-key-input", "value"),
        State("auto-eval-endpoint-url-input", "value"),
        State("auto-eval-model-dropdown", "value"),
        State("auto-eval-openai-api-key-input", "value"),
        State("auto-eval-benchmarks-dropdown", "value"),
    ],
)
def run_evaluation(
    n_clicks, model_source, api_key, endpoint_url, model, openai_api_key, benchmarks
):
    # button_visible_style = {'margin-bottom': '10px', 'display': 'block', 'white-space': 'nowrap'}
    button_hidden_style = {
        "margin-bottom": "10px",
        "display": "none",
        "white-space": "nowrap",
    }

    if not n_clicks:
        return ("", button_hidden_style)
    elif not model_source:
        return ("모델 소스를 선택하세요.", button_hidden_style)

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    elif endpoint_url:
        kwargs["api_base_url"] = endpoint_url

    if not kwargs:
        return ("API Key 또는 엔드포인트 URL을 입력하세요.", button_hidden_style)

    try:
        client = LLMClient(model_source, **kwargs)
        available_models = client.available_models
    except Exception as e:
        return (f"모델을 불러오는 중 오류가 발생했습니다. - {e}", button_hidden_style)

    if not model and model not in available_models:
        return ("모델을 선택하세요.", button_hidden_style)
    elif not benchmarks:
        return ("평가할 벤치마크를 선택하세요.", button_hidden_style)
    elif not ModelSource.OPENAI.is_same(model_source):
        for benchmark in benchmarks:
            if EvaluationMethod.Judge.is_same(
                Benchmark.get_class(benchmark).evaluation_method()
            ):
                if not openai_api_key:
                    return (
                        f"{benchmark} 평가를 위해 OpenAI API Key를 입력하세요.",
                        button_hidden_style,
                    )

                try:
                    client = LLMClient("OpenAI", api_key=openai_api_key)
                except Exception as e:
                    return (
                        f"{benchmark}의 Judge 평가를 위해 OpenAI 모델을 불러오는 중 오류가 발생했습니다. - {e}",
                        button_hidden_style,
                    )

    message = f"평가를 시작합니다...{model}: ["
    for benchmark in benchmarks:
        task_id = shortuuid.uuid()
        task = {
            "created_at": time.time(),
            "task_id": task_id,
            "task_status": EvaluationTaskStatus.Pending.value,
            "model_source": model_source,
            "llm_client_kwargs": kwargs,
            "model": model,
            "benchmark": benchmark,
            "openai_api_key": openai_api_key,
        }
        task_save_path = os.path.join(
            EVAL_TASKS_DIR_PATH, EVAL_TASK_PREFIX + f"{task_id}.json"
        )
        with open(task_save_path, "w") as f:
            f.write(json.dumps(task))

        message += f"{benchmark} - {task_id}, "
    message = message[:-2] + "]"
    return (message, button_hidden_style)


################ Tracking Callback #################


@app.callback(
    Output("tracking-task-cards-container", "children"),
    [
        Input("tracking-refresh-button", "n_clicks"),
        Input("tracking-benchmark-dropdown", "value"),
        Input("tracking-status-dropdown", "value"),
    ],
)
def update_task_cards(n_intervals, selected_benchmark, selected_status):
    tasks = get_evaluation_tasks()
    tasks = sorted(tasks, key=lambda x: x["created_at"], reverse=True)

    task_status2color = {
        EvaluationTaskStatus.Pending.value: "white",
        EvaluationTaskStatus.Started.value: "orange",
        EvaluationTaskStatus.Generating.value: "orange",
        EvaluationTaskStatus.Evaluating.value: "orange",
        EvaluationTaskStatus.Scoring.value: "orange",
        EvaluationTaskStatus.Completed.value: "lightgreen",
        EvaluationTaskStatus.Failed.value: "lightcoral",
    }

    cards = []
    for task in tasks:
        task_id = task.get("task_id")
        task_status = task.get("task_status")
        if any(
            [
                EvaluationTaskStatus.Completed.is_same(selected_status)
                and not EvaluationTaskStatus.Completed.is_same(task_status),
                EvaluationTaskStatus.Failed.is_same(selected_status)
                and not EvaluationTaskStatus.Failed.is_same(task_status),
                selected_status == "Running"
                and EvaluationTaskStatus.Completed.is_same(task_status),
                selected_status == "Running"
                and EvaluationTaskStatus.Failed.is_same(task_status),
            ]
        ):
            continue

        model = task.get("model")
        benchmark = task.get("benchmark")
        if selected_benchmark != "All" and benchmark != selected_benchmark:
            continue

        created_at = task["created_at"]
        date_time = datetime.fromtimestamp(created_at)
        formatted_date = date_time.strftime("%Y-%m-%d %H:%M:%S")

        card_background_color = task_status2color.get(task_status, "white")

        card = dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Div(
                            [
                                html.H4(
                                    benchmark,
                                    className="card-title",
                                    style={
                                        "display": "inline-block",
                                        "margin-right": "20px",
                                    },
                                ),
                                html.P(
                                    model,
                                    className="card-subtitle",
                                    style={"display": "inline-block", "float": "right"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justify-content": "space-between",
                                "align-items": "center",
                            },
                        ),
                        html.Div(
                            [
                                html.P(
                                    f"Task ID: {task_id}",
                                    className="card-text",
                                    style={
                                        "display": "inline-block",
                                        "margin-right": "20px",
                                    },
                                ),
                                html.P(
                                    f"Status: {task_status}",
                                    className="card-text",
                                    style={"display": "inline-block"},
                                ),
                                html.P(
                                    f"Created At: {formatted_date}",
                                    className="card-text",
                                    style={"display": "inline-block"},
                                ),
                                html.P(),
                            ],
                            style={
                                "display": "flex",
                                "justify-content": "space-between",
                                "align-items": "center",
                            },
                        ),
                    ],
                    style={
                        "backgroundColor": card_background_color,
                        "padding-bottom": "0",  # 바닥 여백을 0으로 설정
                        "border": "none",
                        "border-radius": "5px",
                    },
                ),
                className="m-1",
                style={
                    "border-radius": "5px",
                },
            ),
            md=12,
        )
        cards.append(card)
    return cards


@app.callback(
    Output("tracking-refresh-button", "style"),
    Input("tracking-refresh-button", "n_clicks"),
    prevent_initial_call=True,
)
def update_tracking_button_styles(refresh_clicks):
    triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    refresh_style = refresh_button_style.copy()
    if triggered_id == "tracking-refresh-button":
        refresh_style.update(
            {
                "background-color": "#0B6121",
                "color": "slategray",
                "box-shadow": "none",
            }
        )
    return refresh_style


@app.callback(
    Output("tracking-refresh-button", "style", allow_duplicate=True),
    Input("tracking-refresh-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_tracking_button_styles(refresh_clicks):
    refresh_style = refresh_button_style.copy()
    time.sleep(0.5)
    return refresh_style


if __name__ == "__main__":
    monitor_thread = threading.Thread(target=start_monitoring, daemon=True)
    monitor_thread.start()

    app.run_server(debug=True, host="0.0.0.0", port=8080)
