import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from llmgaugeflow.const import Benchmark, ModelSource

dash.register_page(__name__, path="/evaluation")


layout = dbc.Container(
    children=[
        html.H1(
            children="Auto Evaluation",
            style={
                "text-align": "center",
                "color": "slategray",
            },
        ),
        dbc.Row(className="mb-4"),  # 간격 추가
        dbc.Row(
            [
                dbc.Col(
                    dcc.Dropdown(
                        id="auto-eval-model-source-dropdown",
                        options=[
                            {"label": model_source, "value": model_source}
                            for model_source in ModelSource.to_list()
                        ],
                        placeholder="모델 소스 선택",
                        clearable=False,
                        style={
                            "margin-bottom": "10px",
                            "color": "black",
                        },
                    ),
                    md=12,
                ),
                dbc.Col(
                    dbc.Input(
                        id="auto-eval-api-key-input",
                        placeholder="API Key 입력",
                        type="text",
                        style={
                            "margin-bottom": "10px",
                            "display": "none",
                        },
                    ),
                    md=12,
                ),
                dbc.Col(
                    dbc.Input(
                        id="auto-eval-endpoint-url-input",
                        placeholder="모델 엔드포인트 URL 입력",
                        type="text",
                        style={
                            "margin-bottom": "10px",
                            "display": "none",
                        },
                    ),
                    md=12,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="auto-eval-model-dropdown",
                        options=[],
                        placeholder="평가 가능한 모델 선택",
                        clearable=False,
                        style={
                            "margin-bottom": "10px",
                            "color": "black",
                            "display": "none",
                        },
                    ),
                    md=12,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="auto-eval-benchmarks-dropdown",
                        options=[
                            {"label": benchmark, "value": benchmark}
                            for benchmark in Benchmark.to_list()
                        ],
                        multi=True,
                        value=[],
                        placeholder="Choose benchmarks",
                        clearable=False,
                        style={
                            "margin-bottom": "10px",
                            "display": "none",
                        },
                    ),
                    md=12,
                ),
                dbc.Col(
                    dbc.Input(
                        id="auto-eval-openai-api-key-input",
                        placeholder="OpenAI API Key 입력",
                        type="text",
                        style={
                            "margin-bottom": "10px",
                            "display": "none",
                        },
                    ),
                    md=12,
                ),
                dbc.Col(
                    dbc.Button(
                        "평가 시작",
                        id="auto-eval-evaluate-button",
                        color="primary",
                        type="submit",
                        style={
                            "margin-bottom": "10px",
                            "display": "none",
                            "white-space": "nowrap",
                        },
                    ),
                    align="center",
                    style={
                        "width": "20%",
                        "margin": "auto",
                        "margin-top": "20px",
                    },
                    md=12,
                ),
            ],
            className="mb-4",
            align="center",
            style={
                "margin-bottom": "10px",
                "color": "black",
                "width": "50%",
                "margin": "auto",
            },
        ),
        html.Div(
            id="auto-eval-evaluation-output",
            style={"margin-top": "20px", "text-align": "center"},
        ),
    ],
    fluid=True,
    style={
        "padding": "0",
        "margin": "0",
        "height": "100vh",
    },
)
