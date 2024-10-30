import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import dcc, html

from data_utils import get_benchmark_list, get_model_list, update_dataframe
from layout.botton import download_button_style, refresh_button_style
from layout.model_benchmark_selector import detail_model_benchmark_selector

dash.register_page(__name__, path="/detail")


DATAFRAME = update_dataframe()
DETAIL_INITIAL_STORE_DATA = {
    "dataframe": DATAFRAME.to_dict("records"),
    "selected_models": get_model_list(DATAFRAME),
    "selected_benchmark": get_benchmark_list(DATAFRAME, sample=True),
    "selected_prompt_strategy": "All",
    "selected_turn": "All",
}

layout = dbc.Container(
    [
        dcc.Store(
            id="detail-store-selected",
            data=DETAIL_INITIAL_STORE_DATA,
            storage_type="local",
        ),
        detail_model_benchmark_selector,
        dbc.Row(className="mb-4"),  # 간격 추가
        dbc.Row(
            [
                dbc.Col(
                    html.H3(
                        id="detail-benchmark-title",
                        style={
                            "position": "relative",
                            "left": "0",
                            "transform": "translateY(50%)",
                            "white-space": "nowrap",
                        },
                    ),
                    style={"position": "relative", "height": "100%"},
                    md=6,
                ),
                dbc.Col(
                    dbc.Checklist(
                        id="detail-show-all-models",
                        options=[
                            {"label": "Show All Models", "value": True},
                        ],
                        value=[],
                        switch=True,
                        style={
                            "position": "relative",
                            "top": "70%",
                            "white-space": "nowrap",
                        },
                    ),
                    md=2,
                ),
                dbc.Col(
                    id="detail-prompt-strategy-col",
                    children=[
                        dbc.Label("Prompt Strategy", html_for="prompt-strategy"),
                        dcc.Dropdown(
                            id="detail-prompt-strategy",
                            value="All",
                            clearable=False,
                            style={
                                "color": "black",  # 글자색
                                "background-color": "white",  # 배경색
                            },
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    id="detail-turn-col",
                    children=[
                        dbc.Label("Turn", html_for="turn"),
                        dcc.Dropdown(
                            id="detail-turn",
                            value="All",
                            clearable=False,
                            style={
                                "color": "black",  # 글자색
                                "background-color": "white",  # 배경색
                            },
                        ),
                    ],
                    md=2,
                ),
            ],
            className="p-1",
        ),
        dcc.Graph(
            id="detail-graph",
            config={"staticPlot": False, "scrollZoom": False},
            className="p-3",
        ),
        dbc.Row(
            [
                dbc.Col(md=10),
                dbc.Col(
                    dbc.Button(
                        "Download",
                        id="detail-download-button",
                        style=download_button_style,
                        className="p-3",
                    ),
                    md=1,
                ),
                dbc.Col(
                    dbc.Button(
                        "Refresh",
                        id="detail-refresh-button",
                        style=refresh_button_style,
                        className="p-3",
                    ),
                    md=1,
                ),
            ],
            style={
                "color": "black",
                "width": "100%",
                "margin": "auto",
            },
        ),
        dbc.Row(
            [
                dbc.Col(
                    dag.AgGrid(
                        id="detail-grid",
                        defaultColDef={
                            "wrapHeaderText": True,
                            "autoHeaderHeight": True,
                        },
                        dashGridOptions={
                            "domLayout": "normal",
                        },
                        rowClassRules={
                            "bg-secondary text-dark bg-opacity-25": "params.node.rowPinned === 'top' | params.node.rowPinned === 'bottom'"
                        },
                        style={
                            "width": "100%",
                            "height": "100%",
                            "maxHeight": "100%",
                            "overflowY": "auto",  # 세로 스크롤 활성화
                        },
                        columnSize="sizeToFit",
                        className="ag-theme-alpine-dark p-3",
                    ),
                    md=12,
                ),
            ],
            style={
                "margin-bottom": "10px",
                "color": "black",
                "width": "100%",
                "margin": "auto",
                "height": "60%",
                "border": "1px solid #ddd",
                "border-radius": "5px",
                "padding": "5px",
            },
        ),
        dcc.Download(id="detail-download-dataframe"),
    ],
    fluid=True,
    style={
        "padding": "0",
        "margin": "0",
        "height": "100vh",
    },
)
