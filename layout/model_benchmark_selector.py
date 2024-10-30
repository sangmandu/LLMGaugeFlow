import dash_bootstrap_components as dbc
from dash import dcc, html

overall_model_benchmark_selector = dbc.Row(
    dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Button(
                        html.P("Model and Benchmark Selection", className="m-0"),
                        id="overall-model-benchmark-header-btn",
                        className="w-100 p-3 d-flex justify-content-between",
                        color="light",
                        n_clicks=0,
                    ),
                    className="p-0 m-0",
                ),
                dbc.Collapse(
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label(
                                            "Select Model",
                                            html_for="model",
                                        ),
                                        dcc.Dropdown(
                                            id="overall-model",
                                            multi=True,
                                            placeholder="Choose a model",
                                        ),
                                    ],
                                    md=6,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label(
                                            "Select Benchmark",
                                            html_for="benchmark",
                                        ),
                                        dcc.Dropdown(
                                            id="overall-benchmark",
                                            multi=True,
                                            placeholder="Choose a benchmark",
                                        ),
                                    ],
                                    md=6,
                                    sm=12,
                                ),
                            ]
                        ),
                    ),
                    id="overall-model-benchmark-collapse",
                    is_open=False,
                ),
            ]
        )
    ),
    id="overall-model-benchmark-selector",
)

detail_model_benchmark_selector = dbc.Row(
    dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Button(
                        html.P("Model and Benchmark Selection", className="m-0"),
                        id="detail-model-benchmark-header-btn",
                        className="w-100 p-3 d-flex justify-content-between",
                        color="light",
                        n_clicks=0,
                    ),
                    className="p-0 m-0",
                ),
                dbc.Collapse(
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label(
                                            "Select Model",
                                            html_for="model",
                                        ),
                                        dcc.Dropdown(
                                            id="detail-model",
                                            multi=True,
                                            placeholder="Choose a model",
                                        ),
                                    ],
                                    md=6,
                                    sm=12,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label(
                                            "Select Benchmark",
                                            html_for="benchmark",
                                        ),
                                        dcc.Dropdown(
                                            id="detail-benchmark",
                                            placeholder="Choose a benchmark",
                                            clearable=False,
                                        ),
                                    ],
                                    md=6,
                                    sm=12,
                                ),
                            ]
                        ),
                    ),
                    id="detail-model-benchmark-collapse",
                    is_open=False,
                ),
            ]
        )
    ),
    id="detail-model-benchmark-selector",
)
