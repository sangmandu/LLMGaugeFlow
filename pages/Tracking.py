import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from layout.botton import refresh_button_style
from llmgaugeflow.const import Benchmark, EvaluationTaskStatus

dash.register_page(__name__, path="/tracking")


layout = dbc.Container(
    [
        html.H1(
            "Model Evaluation Status Tracker",
            style={
                "text-align": "center",
                "color": "slategray",
            },
        ),
        dbc.Row(className="mb-4"),  # 간격 추가
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="tracking-benchmark-dropdown",
                                        options=[{"label": "All", "value": "All"}]
                                        + [
                                            {"label": bm, "value": bm}
                                            for bm in Benchmark.to_list()
                                        ],
                                        value="All",
                                        clearable=False,
                                        placeholder="Select a Benchmark",
                                        style={
                                            "padding": "0",
                                            "width": "100%",
                                            "margin": "0",
                                        },
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="tracking-status-dropdown",
                                        options=[
                                            {"label": "All", "value": "All"},
                                            {"label": "Running", "value": "Running"},
                                            {
                                                "label": EvaluationTaskStatus.Completed.value,
                                                "value": EvaluationTaskStatus.Completed.value,
                                            },
                                            {
                                                "label": EvaluationTaskStatus.Failed.value,
                                                "value": EvaluationTaskStatus.Failed.value,
                                            },
                                        ],
                                        value="All",
                                        clearable=False,
                                        placeholder="Select a Status",
                                        style={
                                            "padding": "0",
                                            "width": "100%",
                                            "margin": "0",
                                        },
                                    ),
                                    md=6,
                                ),
                            ]
                        )
                    ],
                    md=11,
                ),
                dbc.Col(
                    dbc.Button(
                        "Refresh",
                        id="tracking-refresh-button",
                        style=refresh_button_style,
                        className="p-2",
                    ),
                    md=1,
                ),
            ],
            className="mt-1",
            style={
                "margin-bottom": "10px",
                "color": "black",
                "width": "80%",
                "margin": "auto",
                # "display": "flex",
                # "justify-content": "space-between",
                # "align-items": "center",
            },
        ),
        dbc.Row(
            id="tracking-task-cards-container",
            className="mt-1",
            style={
                "margin-bottom": "10px",
                "color": "black",
                "width": "80%",
                "margin": "auto",
                "max-height": "90vh",  # 최대 높이를 화면의 80%로 설정
                "overflow-y": "scroll",  # 스크롤 활성화
                "border": "1px solid #ddd",  # 테두리 추가
                "border-radius": "5px",  # 테두리 모서리 둥글게
                "padding": "5px",  # 테두리와 내용 간 여백
            },
        ),
    ],
    fluid=True,
    style={
        "padding": "0",
        "margin": "0",
        "height": "100vh",
    },
)
