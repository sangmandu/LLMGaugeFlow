import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from dash import dcc, html

from data_utils import get_benchmark_list, get_model_list, update_dataframe
from layout.botton import download_button_style, refresh_button_style
from layout.model_benchmark_selector import overall_model_benchmark_selector

dash.register_page(__name__, path="/")


DATAFRAME = update_dataframe()
OVERALL_INITIAL_STORE_DATA = {
    "dataframe": DATAFRAME.to_dict("records"),
    "selected_models": get_model_list(DATAFRAME),
    "selected_benchmarks": get_benchmark_list(DATAFRAME),
}

layout = dbc.Container(
    [
        dcc.Store(
            id="overall-store-selected",
            data=OVERALL_INITIAL_STORE_DATA,
            storage_type="local",
        ),
        overall_model_benchmark_selector,
        html.H1(
            children="Overall Evaluation",
            style={
                "text-align": "center",
                "color": "slategray",
            },
            className="p-3",
        ),
        dbc.Row(
            [
                dbc.Col(md=10),
                dbc.Col(
                    dbc.Button(
                        "Download",
                        id="overall-download-button",
                        style=download_button_style,
                        className="p-3",
                    ),
                    md=1,
                ),
                dbc.Col(
                    dbc.Button(
                        "Refresh",
                        id="overall-refresh-button",
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
                        id="overall-grid",
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
                "height": "80%",
                "border": "1px solid #ddd",  # 테두리 추가
                "border-radius": "5px",  # 테두리 모서리 둥글게
                "padding": "5px",  # 테두리와 내용 간 여백
            },
        ),
        dcc.Download(id="overall-download-dataframe"),
    ],
    fluid=True,
    style={
        "padding": "0",
        "margin": "0",
        "height": "100vh",
    },
)
