"""
app.py — The AI Race dashboard.

Layout structure (Cosmograph-inspired):
1. Hero section — big title, centered, space for Spline animation
2. Main content — KPI cards + chart cards (heatmap, pricing scatter)
3. Footer — data attribution
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
from data.fetch import (get_benchmarks, get_openrouter_models, get_hf_top_models,
                        get_notable_models)
from data.process import (build_heatmap_data, build_pricing_data, build_downloads_data,
                          build_timeline_data, build_capabilities_data)

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

# Loading skeleton
loading_skeleton = html.Div(
    className="loading-skeleton",
    children=[html.Div(className="skeleton-row") for _ in range(8)],
)

# ── Layout ──────────────────────────────────────────────────────────
app.layout = html.Div(
    className="dashboard",
    children=[
        # ── Hero ────────────────────────────────────────────────
        html.Section(
            className="hero",
            children=[
                html.Div(
                    className="hero-animation-slot",
                    children=[
                        html.Iframe(
                            src="https://my.spline.design/3drobotheadtrackingmouse-wqtGnyFtfO8RqZnHCwFb0Xaq/",
                            style={
                                "width": "100%",
                                "height": "100%",
                                "border": "none",
                            },
                        ),
                    ],
                ),
                html.Div(
                    className="hero-content",
                    children=[
                        html.H1("The AI Race", className="title"),
                        html.P(
                            "Who's actually winning? A live, data-driven look "
                            "at the top AI models across what matters most.",
                            className="subtitle",
                        ),
                    ],
                ),
            ],
        ),
        # ── Main content ────────────────────────────────────────
        html.Main(
            className="main-content",
            children=[
                # ── Chart 1: Benchmark heatmap ──────────────────
                html.Div(
                    className="chart-section",
                    children=[
                        html.H2(
                            "Which AI model is the best right now?",
                            className="chart-title",
                        ),
                        html.P(id="chart-insight", className="chart-insight"),
                        html.Div(id="kpi-row", className="kpi-row"),
                        dcc.Loading(
                            type="default",
                            color="#9CA3AF",
                            children=[
                                html.Div(
                                    id="heatmap-container",
                                    children=[loading_skeleton],
                                )
                            ],
                        ),
                        html.P(id="last-updated", className="last-updated"),
                    ],
                ),

                # ── Chart 2: Pricing scatter ────────────────────
                html.Div(
                    className="chart-section",
                    style={"marginTop": "2rem"},
                    children=[
                        html.H2(
                            "How much does AI cost and who's cheapest?",
                            className="chart-title",
                        ),
                        html.P(
                            id="pricing-insight",
                            className="chart-insight",
                        ),
                        dcc.Loading(
                            type="default",
                            color="#9CA3AF",
                            children=[
                                html.Div(
                                    id="pricing-container",
                                    children=[loading_skeleton],
                                )
                            ],
                        ),
                        html.P(
                            id="pricing-updated",
                            className="last-updated",
                        ),
                    ],
                ),

                # ── Chart 3: Downloads by org ───────────────────
                html.Div(
                    className="chart-section",
                    style={"marginTop": "2rem"},
                    children=[
                        html.H2(
                            "Who's winning the open-source AI race?",
                            className="chart-title",
                        ),
                        html.P(
                            id="downloads-insight",
                            className="chart-insight",
                        ),
                        dcc.Loading(
                            type="default",
                            color="#9CA3AF",
                            children=[
                                html.Div(
                                    id="downloads-container",
                                    children=[loading_skeleton],
                                )
                            ],
                        ),
                        html.P(
                            id="downloads-updated",
                            className="last-updated",
                        ),
                    ],
                ),

                # ── Chart 4: Model release timeline ─────────────
                html.Div(
                    className="chart-section",
                    style={"marginTop": "2rem"},
                    children=[
                        html.H2(
                            "Who's investing most in winning this race?",
                            className="chart-title",
                        ),
                        html.P(
                            id="timeline-insight",
                            className="chart-insight",
                        ),
                        dcc.Loading(
                            type="default",
                            color="#9CA3AF",
                            children=[
                                html.Div(
                                    id="timeline-container",
                                    children=[loading_skeleton],
                                )
                            ],
                        ),
                        html.P(
                            id="timeline-updated",
                            className="last-updated",
                        ),
                    ],
                ),

                # ── Chart 5: Capabilities heatmap ───────────────
                html.Div(
                    className="chart-section",
                    style={"marginTop": "2rem"},
                    children=[
                        html.H2(
                            "What can each company's AI actually do?",
                            className="chart-title",
                        ),
                        html.P(
                            id="capabilities-insight",
                            className="chart-insight",
                        ),
                        dcc.Loading(
                            type="default",
                            color="#9CA3AF",
                            children=[
                                html.Div(
                                    id="capabilities-container",
                                    children=[loading_skeleton],
                                )
                            ],
                        ),
                        html.P(
                            id="capabilities-updated",
                            className="last-updated",
                        ),
                    ],
                ),

            ],
        ),
        # ── Footer ──────────────────────────────────────────────
        html.Footer(
            className="footer",
            children=[
                html.P(
                    [
                        "Data: ",
                        html.A("Epoch AI", href="https://epoch.ai",
                               target="_blank"),
                        " (CC BY 4.0)  ·  ",
                        html.A("OpenRouter", href="https://openrouter.ai",
                               target="_blank"),
                        "  ·  ",
                        html.A("HuggingFace", href="https://huggingface.co",
                               target="_blank"),
                        "  ·  Built with Plotly Dash",
                    ]
                ),
            ],
        ),
        # Refresh triggers — separate intervals for different cache durations
        dcc.Interval(id="refresh-24h", interval=86400 * 1000, n_intervals=0),
        dcc.Interval(id="refresh-5min", interval=300 * 1000, n_intervals=0),
    ],
)


# ── Callback 1: Benchmark heatmap ──────────────────────────────────
@app.callback(
    Output("heatmap-container", "children"),
    Output("chart-insight", "children"),
    Output("last-updated", "children"),
    Output("kpi-row", "children"),
    Input("refresh-24h", "n_intervals"),
)
def update_heatmap(_n):
    try:
        raw_df = get_benchmarks()
        heatmap_df, last_date = build_heatmap_data(raw_df)
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text=f"Could not load data: {e}",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#EF4444"))],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True})
        return graph, "Data unavailable.", "", []

    models = heatmap_df.index.tolist()
    categories = heatmap_df.columns.tolist()
    scores = heatmap_df.values

    # Truncate long model names for mobile readability
    def truncate(name, max_len=22):
        return name if len(name) <= max_len else name[:max_len-1] + "…"

    ranked_models = [f"#{i+1}  {truncate(m)}" for i, m in enumerate(models)]

    # Column winners
    col_max_idx = {}
    for j, cat in enumerate(categories):
        valid = [(i, v) for i, v in enumerate(scores[:, j]) if pd.notna(v)]
        if valid:
            col_max_idx[j] = max(valid, key=lambda x: x[1])[0]

    # Hover text
    hover_text = []
    for i, model in enumerate(models):
        row = []
        for j, cat in enumerate(categories):
            val = scores[i][j]
            if pd.notna(val):
                row.append(f"{model}<br>{cat}: {val:.1f}%")
            else:
                row.append(f"{model}<br>{cat}: No data")
        hover_text.append(row)

    # Cell text with winner stars
    cell_text = []
    for i in range(len(models)):
        row = []
        for j in range(len(categories)):
            val = scores[i][j]
            if pd.notna(val):
                star = " ★" if col_max_idx.get(j) == i else ""
                row.append(f"{val:.1f}{star}")
            else:
                row.append("—")
        cell_text.append(row)

    # Normalize for dynamic text color
    all_valid = [v for row in scores for v in row if pd.notna(v)]
    s_min, s_max = min(all_valid), max(all_valid)

    fig = go.Figure(
        data=go.Heatmap(
            z=scores, x=categories, y=ranked_models,
            hovertext=hover_text,
            hovertemplate="%{hovertext}<extra></extra>",
            colorscale=[
                [0.0, "#111318"], [0.2, "#1B2B4B"],
                [0.45, "#374151"], [0.65, "#9CA3AF"],
                [0.85, "#D1D5DB"], [1.0, "#F3F4F6"],
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="Score %", font=dict(color="#6B7280", size=10)),
                tickfont=dict(color="#6B7280", size=10),
                bgcolor="rgba(0,0,0,0)", borderwidth=0, len=0.75,
            ),
            xgap=3, ygap=3,
        )
    )

    # Annotations for per-cell text color
    annotations = []
    for i in range(len(models)):
        for j in range(len(categories)):
            val = scores[i][j]
            if pd.notna(val):
                norm = (val - s_min) / (s_max - s_min) if s_max > s_min else 0.5
                text_color = "#111318" if norm > 0.6 else "#F3F4F6"
                label = cell_text[i][j]
            else:
                text_color = "#4B5563"
                label = "—"
            annotations.append(dict(
                x=categories[j], y=ranked_models[i], text=label,
                showarrow=False,
                font=dict(size=12, color=text_color, family="IBM Plex Mono, monospace"),
            ))

    fig.update_layout(
        annotations=annotations,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D1D5DB", family="Outfit, system-ui, sans-serif"),
        yaxis=dict(autorange="reversed",
                   tickfont=dict(size=11.5, family="IBM Plex Mono, monospace", color="#D1D5DB"),
                   side="left"),
        xaxis=dict(tickfont=dict(size=12, color="#9CA3AF"), side="top"),
        margin=dict(l=160, r=30, t=35, b=15),
        height=400,
        autosize=True,
    )

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True})

    # Insight
    top_model = models[0]
    top_scores = heatmap_df.loc[top_model]
    best_cat = top_scores.idxmax()
    best_val = top_scores.max()
    insight = (
        f"{top_model} leads overall, scoring highest in {best_cat} "
        f"({best_val:.1f}%). No single model dominates every category "
        f"— the ★ marks each column's winner."
    )

    updated_text = f"Epoch AI  ·  {last_date}"

    kpi_cards = [
        html.Div(className="kpi-card", children=[
            html.Div(str(len(models)), className="kpi-value"),
            html.Div("Models Ranked", className="kpi-label"),
        ]),
        html.Div(className="kpi-card", children=[
            html.Div(str(len(categories)), className="kpi-value"),
            html.Div("Categories", className="kpi-label"),
        ]),
        html.Div(className="kpi-card", children=[
            html.Div(str(last_date), className="kpi-value"),
            html.Div("Last Data Update", className="kpi-label"),
        ]),
    ]

    return graph, insight, updated_text, kpi_cards


# ── Callback 2: Pricing scatter ────────────────────────────────────
# Org → color mapping (navy palette + distinct accents)
ORG_COLORS = {
    "openai": "#E5E7EB",      # light gray
    "anthropic": "#9CA3AF",   # mid gray
    "google": "#6B7280",      # gray
    "meta-llama": "#4B5563",  # dark gray
    "mistralai": "#D1D5DB",   # silver
    "deepseek": "#374151",    # charcoal
    "x-ai": "#F3F4F6",        # off-white
    "qwen": "#1B2B4B",        # navy
    "Other": "#1F2937",       # dark slate
}


@app.callback(
    Output("pricing-container", "children"),
    Output("pricing-insight", "children"),
    Output("pricing-updated", "children"),
    Input("refresh-5min", "n_intervals"),
)
def update_pricing(_n):
    try:
        raw_models = get_openrouter_models()
        df = build_pricing_data(raw_models)
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text=f"Could not load pricing: {e}",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#EF4444"))],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True})
        return graph, "Data unavailable.", ""

    fig = go.Figure()

    # Compute dot sizes from context_length using log scale.
    # Context ranges from ~4K to ~1M+ tokens — log compresses this
    # into a usable visual range (min 5px, max 28px).
    import numpy as np
    ctx = df["context_length"].clip(lower=1)  # avoid log(0)
    log_ctx = np.log10(ctx)
    log_min, log_max = log_ctx.min(), log_ctx.max()
    size_min, size_max = 5, 28
    if log_max > log_min:
        df["dot_size"] = size_min + (log_ctx - log_min) / (log_max - log_min) * (size_max - size_min)
    else:
        df["dot_size"] = (size_min + size_max) / 2

    # One trace per org so each gets its own legend entry + color
    plot_order = [o for o in ORG_COLORS if o != "Other"]

    for org in plot_order:
        subset = df[df["org_display"] == org]
        if subset.empty:
            continue

        fig.add_trace(go.Scatter(
            x=subset["prompt_1m"],
            y=subset["completion_1m"],
            mode="markers",
            name=org,
            marker=dict(
                color=ORG_COLORS.get(org, "#334155"),
                size=subset["dot_size"],
                opacity=0.85,
                line=dict(width=0),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Prompt: $%{x:.2f}/1M tokens<br>"
                "Completion: $%{y:.2f}/1M tokens<br>"
                "Context: %{customdata[1]:,} tokens"
                "<extra>%{customdata[2]}</extra>"
            ),
            customdata=list(zip(
                subset["name"],
                subset["context_length"],
                subset["org"],
            )),
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D1D5DB", family="Outfit, system-ui, sans-serif"),
        xaxis=dict(
            title=dict(text="Prompt price ($ per 1M tokens)",
                       font=dict(size=11, color="#9CA3AF")),
            tickfont=dict(size=10, color="#6B7280"),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            type="log",
        ),
        yaxis=dict(
            title=dict(text="Completion price ($ per 1M tokens)",
                       font=dict(size=11, color="#9CA3AF")),
            tickfont=dict(size=10, color="#6B7280"),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            type="log",
        ),
        legend=dict(
            font=dict(size=11, color="#D1D5DB"),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=50, r=15, t=40, b=45),
        height=420,
        autosize=True,
    )

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True})

    # Insight: find the cheapest model overall (lowest prompt + completion)
    df["total_1m"] = df["prompt_1m"] + df["completion_1m"]
    cheapest = df.loc[df["total_1m"].idxmin()]
    total_models = len(df)

    insight = (
        f"{total_models} models compared. "
        f"Cheapest overall: {cheapest['name']} "
        f"(${cheapest['prompt_1m']:.2f} prompt + "
        f"${cheapest['completion_1m']:.2f} completion per 1M tokens). "
        f"Bigger dots = larger context window. Log scale."
    )

    updated_text = "OpenRouter API  ·  Live"

    return graph, insight, updated_text


# ── Callback 3: Downloads by org ───────────────────────────────────
# Color mapping — reuse org colors where possible, add new ones
DOWNLOAD_COLORS = {
    "Qwen (Alibaba)": "#E5E7EB",
    "Meta": "#9CA3AF",
    "OpenAI (community)": "#D1D5DB",
    "OpenAI": "#D1D5DB",
    "DeepSeek": "#6B7280",
    "NVIDIA": "#4B5563",
    "Microsoft": "#374151",
    "Mistral": "#F3F4F6",
    "Google": "#6B7280",
    "Meta (legacy)": "#9CA3AF",
    "EleutherAI": "#1B2B4B",
    "HuggingFace": "#4B5563",
}


@app.callback(
    Output("downloads-container", "children"),
    Output("downloads-insight", "children"),
    Output("downloads-updated", "children"),
    Input("refresh-5min", "n_intervals"),
)
def update_downloads(_n):
    try:
        raw_models = get_hf_top_models()
        df = build_downloads_data(raw_models, top_n=12)
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text=f"Could not load downloads: {e}",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#EF4444"))],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True})
        return graph, "Data unavailable.", ""

    # Reverse for horizontal bar chart (highest at top)
    df = df.sort_values("downloads", ascending=True)

    # Format downloads for display (e.g., 138M, 31M)
    def fmt(n):
        if n >= 1_000_000:
            return f"{n / 1_000_000:.0f}M"
        if n >= 1_000:
            return f"{n / 1_000:.0f}K"
        return str(n)

    bar_colors = [DOWNLOAD_COLORS.get(name, "#374151") for name in df["display_name"]]

    fig = go.Figure(
        data=go.Bar(
            x=df["downloads"],
            y=df["display_name"],
            orientation="h",
            marker=dict(
                color=bar_colors,
                cornerradius=4,
            ),
            text=[fmt(d) for d in df["downloads"]],
            textposition="outside",
            textfont=dict(size=11, color="#D1D5DB", family="IBM Plex Mono, monospace"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Downloads: %{x:,.0f}<br>"
                "Models in top 200: %{customdata}"
                "<extra></extra>"
            ),
            customdata=df["model_count"],
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D1D5DB", family="Outfit, system-ui, sans-serif"),
        xaxis=dict(
            visible=False,
            # Extra room for the text labels outside bars
            range=[0, df["downloads"].max() * 1.25],
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#D1D5DB"),
        ),
        margin=dict(l=120, r=50, t=10, b=10),
        height=400,
        autosize=True,
        bargap=0.25,
    )

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True})

    # Insight
    top_org = df.iloc[-1]  # last row is highest (we sorted ascending)
    total_downloads = df["downloads"].sum()

    insight = (
        f"{top_org['display_name']} leads with {fmt(top_org['downloads'])} downloads "
        f"across {top_org['model_count']} models in the top 200. "
        f"Total across top 12 orgs: {fmt(total_downloads)}. "
        f"Based on HuggingFace text-generation models."
    )

    updated_text = "HuggingFace API  ·  Live"

    return graph, insight, updated_text


# ── Callback 4: Model release timeline ─────────────────────────────
TIMELINE_COLORS = {
    "Google": "#E5E7EB",
    "OpenAI": "#D1D5DB",
    "Meta": "#9CA3AF",
    "Anthropic": "#6B7280",
    "Alibaba": "#4B5563",
    "Microsoft": "#F3F4F6",
    "DeepSeek": "#374151",
    "xAI": "#D1D5DB",
    "Baidu": "#1B2B4B",
    "ByteDance": "#6B7280",
    "NVIDIA": "#9CA3AF",
    "Mistral": "#E5E7EB",
}


@app.callback(
    Output("timeline-container", "children"),
    Output("timeline-insight", "children"),
    Output("timeline-updated", "children"),
    Input("refresh-24h", "n_intervals"),
)
def update_timeline(_n):
    try:
        raw_df = get_notable_models()
        df = build_timeline_data(raw_df)
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text=f"Could not load timeline: {e}",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#EF4444"))],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True})
        return graph, "Data unavailable.", ""

    import numpy as np

    fig = go.Figure()

    # Only show models with compute data for the Y axis to be meaningful
    df_with_compute = df[df["has_compute"]].copy()

    # Plot each org as its own trace
    for org in TIMELINE_COLORS:
        subset = df_with_compute[df_with_compute["org"] == org]
        if subset.empty:
            continue

        # Dot size from parameters (log scale, 6-22px range)
        params = subset["parameters"].clip(lower=1)
        log_p = np.log10(params.fillna(params.median()))
        p_min, p_max = log_p.min(), log_p.max()
        if p_max > p_min:
            sizes = 6 + (log_p - p_min) / (p_max - p_min) * 16
        else:
            sizes = 12

        fig.add_trace(go.Scatter(
            x=subset["date"],
            y=subset["compute_flop"],
            mode="markers",
            name=org,
            marker=dict(
                color=TIMELINE_COLORS[org],
                size=sizes,
                opacity=0.8,
                line=dict(width=0),
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Date: %{x|%b %Y}<br>"
                "Compute: %{y:.1e} FLOP"
                "<extra></extra>"
            ),
            customdata=list(zip(subset["model"], subset["org"])),
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D1D5DB", family="Outfit, system-ui, sans-serif"),
        xaxis=dict(
            title=dict(text="Publication date",
                       font=dict(size=11, color="#9CA3AF")),
            tickfont=dict(size=10, color="#6B7280"),
            gridcolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(
            title=dict(text="Training compute (FLOP)",
                       font=dict(size=11, color="#9CA3AF")),
            tickfont=dict(size=10, color="#6B7280"),
            gridcolor="rgba(255,255,255,0.06)",
            type="log",
        ),
        legend=dict(
            font=dict(size=11, color="#D1D5DB"),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
        ),
        margin=dict(l=60, r=15, t=40, b=45),
        height=450,
        autosize=True,
    )

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True})

    # Insight: who has the most models, who has the biggest compute
    org_counts = df["org"].value_counts()
    most_active = org_counts.index[0]
    most_active_count = org_counts.iloc[0]

    if not df_with_compute.empty:
        biggest = df_with_compute.loc[df_with_compute["compute_flop"].idxmax()]
        insight = (
            f"Since 2020, {most_active} has released {most_active_count} notable models "
            f"— more than anyone else. The largest training run recorded: "
            f"{biggest['model']} by {biggest['org']} "
            f"({biggest['compute_flop']:.1e} FLOP). "
            f"Bigger dots = more parameters."
        )
    else:
        insight = f"{most_active} leads with {most_active_count} notable models since 2020."

    updated_text = "Epoch AI notable models  ·  Updated daily"

    return graph, insight, updated_text


# ── Callback 5: Capabilities heatmap ───────────────────────────────
@app.callback(
    Output("capabilities-container", "children"),
    Output("capabilities-insight", "children"),
    Output("capabilities-updated", "children"),
    Input("refresh-5min", "n_intervals"),
)
def update_capabilities(_n):
    try:
        raw = get_openrouter_models()
        df = build_capabilities_data(raw, top_n=10)
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text=f"Could not load capabilities: {e}",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#EF4444"))],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True})
        return graph, "Data unavailable.", ""

    from data.process import CAPABILITY_COLUMNS

    orgs = df["display_name"].tolist()
    caps = CAPABILITY_COLUMNS
    z = df[caps].values

    # Dynamic text color: dark on bright cells, light on dark cells
    z_max = z.max() if z.max() > 0 else 1

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=caps,
            y=orgs,
            colorscale=[
                [0.0, "#111318"],
                [0.01, "#1B2B4B"],
                [0.2, "#374151"],
                [0.45, "#6B7280"],
                [0.7, "#D1D5DB"],
                [1.0, "#F3F4F6"],
            ],
            showscale=False,
            xgap=3,
            ygap=3,
        )
    )

    # Annotations with dynamic text color
    annotations = []
    for i, org in enumerate(orgs):
        for j, cap in enumerate(caps):
            val = int(z[i][j])
            norm = val / z_max if z_max > 0 else 0
            text_color = "#111318" if norm > 0.5 else "#F3F4F6"
            if val == 0:
                text_color = "#374151"
            annotations.append(dict(
                x=cap, y=org, text=str(val),
                showarrow=False,
                font=dict(size=12, color=text_color, family="IBM Plex Mono, monospace"),
            ))

    fig.update_layout(
        annotations=annotations,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D1D5DB", family="Outfit, system-ui, sans-serif"),
        yaxis=dict(
            tickfont=dict(size=12, color="#D1D5DB"),
            side="left",
            autorange="reversed",
        ),
        xaxis=dict(
            tickfont=dict(size=11, color="#9CA3AF"),
            side="top",
        ),
        margin=dict(l=90, r=15, t=35, b=15),
        height=400,
        autosize=True,
    )

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True})

    # Insight: find most versatile org (most non-zero capabilities)
    df["num_caps"] = (df[caps] > 0).sum(axis=1)
    most_versatile = df.loc[df["num_caps"].idxmax()]
    text_only = df[df["num_caps"] == 0]["display_name"].tolist()

    insight = (
        f"{most_versatile['display_name']} is the most versatile with "
        f"{int(most_versatile['num_caps'])} capability types across "
        f"{most_versatile['total_models']} models."
    )
    if text_only:
        insight += f" {', '.join(text_only)}: text only — no multimodal capabilities."

    updated_text = "OpenRouter API  ·  Live"

    return graph, insight, updated_text


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
