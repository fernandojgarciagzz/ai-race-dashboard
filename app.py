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
from datetime import datetime, timezone
from data.fetch import (get_benchmarks, get_openrouter_models, get_hf_top_models,
                        get_notable_models, get_benchmarks_models, get_gpu_clusters,
                        get_ml_hardware, get_last_fetch_time,
                        get_hf_trending, get_hf_modality_champions,
                        get_swebench_leaderboard)
from data.process import (build_heatmap_data, build_pricing_data, build_downloads_data,
                          build_timeline_data, filter_flagship_models,
                          build_chip_dominance_data, build_context_data,
                          build_gpu_map_data,
                          build_trending_data, build_modality_champions,
                          build_swebench_data)

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.index_string = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    {%metas%}
    <title>The AI Race — Live Dashboard</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏁</text></svg>">
    <meta property="og:type" content="website">
    <meta property="og:title" content="The AI Race — Who's Actually Winning?">
    <meta property="og:description" content="Live, data-driven dashboard tracking AI models across benchmarks, pricing, downloads, compute, and capabilities.">
    <meta property="og:url" content="https://ai-race-dashboard.onrender.com">
    <meta name="twitter:card" content="summary">
    <meta name="twitter:title" content="The AI Race — Who's Actually Winning?">
    <meta name="twitter:description" content="Live dashboard: benchmarks, pricing, downloads, compute trends for top AI models.">
    <script>if(localStorage.getItem('ai-dash-theme')==='dark')document.documentElement.classList.add('dark');</script>
    {%css%}
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
    <script>
    (function() {
        var observer = new MutationObserver(function() {
            var toggle = document.getElementById('theme-toggle');
            if (toggle && !toggle._bound) {
                toggle._bound = true;
                var icon = document.getElementById('theme-icon');
                if (document.documentElement.classList.contains('dark') && icon) icon.textContent = '☾';
                toggle.addEventListener('click', function() {
                    var isDark = document.documentElement.classList.toggle('dark');
                    localStorage.setItem('ai-dash-theme', isDark ? 'dark' : 'light');
                    if (icon) icon.textContent = isDark ? '☾' : '☀';
                });
            }
            var mobileBtn = document.getElementById('mobile-toggle');
            if (mobileBtn && !mobileBtn._bound) {
                mobileBtn._bound = true;
                mobileBtn.addEventListener('click', function() {
                    document.getElementById('nav-links').classList.toggle('open');
                    mobileBtn.classList.toggle('active');
                });
            }
            var nav = document.getElementById('main-nav');
            if (nav && !window._navScrollBound) {
                window._navScrollBound = true;
                var hero = document.querySelector('.hero');
                var heroContent = document.querySelector('.hero-content');
                window.addEventListener('scroll', function() {
                    if (!hero) return;
                    nav.classList.toggle('visible', hero.getBoundingClientRect().bottom <= 0);
                    if (heroContent) {
                        var progress = Math.min(window.scrollY / (window.innerHeight * 0.5), 1);
                        heroContent.style.transform = 'translateY(' + (-progress * 120) + 'px)';
                        heroContent.style.opacity = 1 - progress;
                    }
                }, {passive: true});
            }
            var sections = document.querySelectorAll('[id$="-container"]');
            var navLinks = document.querySelectorAll('.nav-link');
            if (sections.length && navLinks.length && !window._ioSetup) {
                window._ioSetup = true;
                var io = new IntersectionObserver(function(entries) {
                    entries.forEach(function(entry) {
                        if (entry.isIntersecting) {
                            navLinks.forEach(function(l) { l.classList.remove('active'); });
                            var t = document.querySelector('.nav-link[href="#' + entry.target.id + '"]');
                            if (t) t.classList.add('active');
                        }
                    });
                }, {rootMargin: '-30% 0px -60% 0px'});
                sections.forEach(function(s) { io.observe(s); });
            }
            if (navLinks.length && !window._scrollBound) {
                window._scrollBound = true;
                navLinks.forEach(function(link) {
                    link.addEventListener('click', function(e) {
                        e.preventDefault();
                        var target = document.getElementById(this.getAttribute('href').slice(1));
                        if (target) target.scrollIntoView({behavior: 'smooth', block: 'start'});
                        document.getElementById('nav-links').classList.remove('open');
                        var mb = document.getElementById('mobile-toggle');
                        if (mb) mb.classList.remove('active');
                    });
                });
            }
        });
        observer.observe(document.body, {childList: true, subtree: true});
    })();
    </script>
</body>
</html>'''

# ── Master company color palette ──────────────────────────────────
COMPANY_COLORS = {
    "OpenAI": "#EEEEEE",
    "Anthropic": "#E8906A",
    "Google": "#4285F4",
    "Meta": "#0099FF",
    "DeepSeek": "#4D6BFE",
    "Mistral": "#FF7000",
    "xAI": "#A8B5C8",
    "Alibaba": "#7C3AED",
    "Microsoft": "#00A4EF",
    "NVIDIA": "#76B900",
    "Baidu": "#DE0A23",
    "ByteDance": "#00D4AA",
    "Moonshot": "#EC4899",
    "Zhipu AI": "#6366F1",
    "MiniMax": "#94A3B8",
    "Other": "#4B5563",
}


def _format_updated(source_label: str, source_key: str) -> str:
    ts = get_last_fetch_time(source_key)
    if ts == 0:
        return f"{source_label}  ·  Fetching..."
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"{source_label}  ·  Last refreshed: {dt.strftime('%Y-%m-%d %H:%M')} UTC"


def _fmt_number(n):
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


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
        # ── Floating portfolio link ───────────────────────────
        html.A(
            "← fernandogarciag.com",
            href="https://fernandogarciag.com",
            className="portfolio-link",
            target="_blank",
        ),
        # ── Navigation ─────────────────────────────────────────
        html.Nav(
            id="main-nav",
            className="site-nav",
            children=[
                html.Div(
                    className="nav-inner",
                    children=[
                        html.A(
                            "fernandogarciag",
                            href="https://fernandogarciag.com",
                            className="nav-brand",
                            target="_blank",
                        ),
                        html.Button(
                            html.Span(className="hamburger-icon"),
                            id="mobile-toggle",
                            className="mobile-toggle",
                            **{"aria-label": "Toggle navigation"},
                        ),
                        html.Div(
                            className="nav-right",
                            children=[
                                html.Div(
                                    className="nav-links",
                                    id="nav-links",
                                    children=[
                                        html.A("Benchmarks", href="#heatmap-container", className="nav-link"),
                                        html.A("Context", href="#context-container", className="nav-link"),
                                        html.A("Pricing", href="#pricing-container", className="nav-link"),
                                        html.A("Scaling", href="#timeline-container", className="nav-link"),
                                        html.A("Hardware", href="#chips-container", className="nav-link"),
                                        html.A("Map", href="#gpu-map-container", className="nav-link"),
                                        html.A("Downloads", href="#downloads-container", className="nav-link"),
                                        html.A("Trending", href="#trending-container", className="nav-link"),
                                        html.A("Modality", href="#modality-container", className="nav-link"),
                                        html.A("SWE-bench", href="#swebench-container", className="nav-link"),
                                    ],
                                ),
                                html.Button(
                                    html.Span("☀", id="theme-icon"),
                                    id="theme-toggle",
                                    className="theme-toggle",
                                    **{"aria-label": "Toggle dark mode"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
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
                        html.P(
                            "Top 8 models ranked by average score across 5 specific benchmarks "
                            "(MATH level 5, WeirdML, ARC-AGI-2, GPQA, SimpleBench). "
                            "Each cell shows a percentage — ★ marks the best in each column. "
                            "These reflect narrow benchmark performance, not overall capability.",
                            className="chart-description",
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

                # ── Chart 2: Context window distribution ──────────
                html.Div(
                    className="chart-section",
                    style={"marginTop": "2rem"},
                    children=[
                        html.H2(
                            "How much memory can AI hold?",
                            className="chart-title",
                        ),
                        html.P(
                            "Context window = how much text a model can process at once. "
                            "For reference: 8K tokens ≈ 12 pages, 128K ≈ a full novel, 1M ≈ 10 novels. "
                            "Hover for examples. Advertised maximums — effective context may vary.",
                            className="chart-description",
                        ),
                        html.P(
                            id="context-insight",
                            className="chart-insight",
                        ),
                        dcc.Loading(
                            type="default",
                            color="#9CA3AF",
                            children=[
                                html.Div(
                                    id="context-container",
                                    children=[loading_skeleton],
                                )
                            ],
                        ),
                        html.P(
                            id="context-updated",
                            className="last-updated",
                        ),
                    ],
                ),

                # ── Chart 3: Pricing scatter ────────────────────
                html.Div(
                    className="chart-section",
                    style={"marginTop": "2rem"},
                    children=[
                        html.H2(
                            "How much does AI cost and who's cheapest?",
                            className="chart-title",
                        ),
                        html.P(
                            "Each dot is a model. X = prompt cost, Y = completion cost (per 1M tokens). "
                            "Bigger dots = larger context window. Both axes use log scale.",
                            className="chart-description",
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

                # ── Chart 4: Model release timeline ─────────────
                html.Div(
                    className="chart-section",
                    style={"marginTop": "2rem"},
                    children=[
                        html.H2(
                            "How fast is AI scaling?",
                            className="chart-title",
                        ),
                        html.P(
                            "Every notable AI model since 2020, plotted by release date and training compute (FLOP). "
                            "Higher = more compute spent training. Bigger dots = more parameters. "
                            "◆ diamonds = recent models whose compute hasn't been published yet.",
                            className="chart-description",
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

                # ── Chart 7: AI chip dominance ────────────────────
                html.Div(
                    className="chart-section",
                    style={"marginTop": "2rem"},
                    children=[
                        html.H2(
                            "Who makes the chips powering AI?",
                            className="chart-title",
                        ),
                        html.P(
                            "Every registered AI chip by manufacturer. "
                            "Blue = US company, Red = China, Gray = other. "
                            "Includes GPUs, TPUs, and custom ASICs.",
                            className="chart-description",
                        ),
                        html.P(
                            id="chips-insight",
                            className="chart-insight",
                        ),
                        dcc.Loading(
                            type="default",
                            color="#9CA3AF",
                            children=[
                                html.Div(
                                    id="chips-container",
                                    children=[loading_skeleton],
                                )
                            ],
                        ),
                        html.P(
                            id="chips-updated",
                            className="last-updated",
                        ),
                    ],
                ),

                # ── Chart 8: GPU cluster world map ───────────────
                html.Div(
                    className="chart-section",
                    style={"marginTop": "2rem"},
                    children=[
                        html.H2(
                            "Where is AI compute? A global map",
                            className="chart-title",
                        ),
                        html.P(
                            "Every known GPU cluster in the world, plotted by location. "
                            "Bigger dots = more compute (H100 equivalents). "
                            "Blue = US, Red = China, Teal = Europe. Hover for details.",
                            className="chart-description",
                        ),
                        html.P(
                            id="gpu-map-insight",
                            className="chart-insight",
                        ),
                        dcc.Loading(
                            type="default",
                            color="#9CA3AF",
                            children=[
                                html.Div(
                                    id="gpu-map-container",
                                    children=[loading_skeleton],
                                )
                            ],
                        ),
                        html.P(
                            id="gpu-map-updated",
                            className="last-updated",
                        ),
                    ],
                ),

                # ── Chart 8: SWE-bench leaderboard ────────────────
                html.Div(
                    className="chart-section",
                    style={"marginTop": "2rem"},
                    children=[
                        html.H2(
                            "Which AI can actually code?",
                            className="chart-title",
                        ),
                        html.P(
                            "SWE-bench measures real-world software engineering: given a GitHub issue, "
                            "can the AI generate a correct patch? Higher % = more issues resolved. "
                            "Results depend on submissions — newer models may not yet be evaluated.",
                            className="chart-description",
                        ),
                        html.P(id="swebench-insight", className="chart-insight"),
                        dcc.Loading(
                            type="default",
                            color="#9CA3AF",
                            children=[
                                html.Div(
                                    id="swebench-container",
                                    children=[loading_skeleton],
                                )
                            ],
                        ),
                        html.P(id="swebench-updated", className="last-updated"),
                    ],
                ),

                # ── Open Source section ────────────────────────────
                # ── Downloads by org ─────────────────────────────
                html.Div(
                    className="chart-section",
                    style={"marginTop": "2rem"},
                    children=[
                        html.H2(
                            "Who's winning the open-source AI race?",
                            className="chart-title",
                        ),
                        html.P(
                            "Downloads across HuggingFace's top 200 text-generation models, "
                            "weighted by recency (60-day half-life). Measures current momentum, not historical totals.",
                            className="chart-description",
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

                # ── Trending models ─────────────────────────────
                html.Div(
                    className="chart-section",
                    style={"marginTop": "2rem"},
                    children=[
                        html.H2(
                            "What's trending in open-source AI?",
                            className="chart-title",
                        ),
                        html.P(
                            "Top 15 trending open-source models on HuggingFace right now, "
                            "ranked by trending score. Freshness matters — newer models trend higher.",
                            className="chart-description",
                        ),
                        html.P(id="trending-insight", className="chart-insight"),
                        dcc.Loading(
                            type="default",
                            color="#9CA3AF",
                            children=[
                                html.Div(
                                    id="trending-container",
                                    children=[loading_skeleton],
                                )
                            ],
                        ),
                        html.P(id="trending-updated", className="last-updated"),
                    ],
                ),

                # ── Modality champions ─────────────────────────────
                html.Div(
                    className="chart-section",
                    style={"marginTop": "2rem"},
                    children=[
                        html.H2(
                            "Who leads each open-source AI modality?",
                            className="chart-title",
                        ),
                        html.P(
                            "The #1 open-source model by downloads for each major AI capability: "
                            "image generation, voice synthesis, video, music, animation, 3D, and transcription.",
                            className="chart-description",
                        ),
                        html.P(id="modality-insight", className="chart-insight"),
                        dcc.Loading(
                            type="default",
                            color="#9CA3AF",
                            children=[
                                html.Div(
                                    id="modality-container",
                                    children=[loading_skeleton],
                                )
                            ],
                        ),
                        html.P(id="modality-updated", className="last-updated"),
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
                        "  ·  ",
                        html.A("SWE-bench", href="https://www.swebench.com",
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
        heatmap_df, last_date, total_evaluated, num_dropped = build_heatmap_data(raw_df)
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text=f"Could not load data: {e}",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#EF4444"))],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})
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
                   side="left", fixedrange=True),
        xaxis=dict(tickfont=dict(size=12, color="#9CA3AF"), side="top", fixedrange=True),
        dragmode=False,
        margin=dict(l=160, r=30, t=35, b=15),
        height=400,
        autosize=True,
    )

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})

    # Insight
    top_model = models[0]
    top_scores = heatmap_df.loc[top_model]
    best_cat = top_scores.idxmax()
    best_val = top_scores.max()
    insight = (
        f"{top_model} leads overall, scoring highest in {best_cat} "
        f"({best_val:.1f}%). "
        f"{len(models)} models ranked from {total_evaluated} evaluated "
        f"({num_dropped} had insufficient data)."
    )

    updated_text = _format_updated("Epoch AI", "benchmarks")

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
ORG_COLORS = {
    "openai": COMPANY_COLORS["OpenAI"],
    "anthropic": COMPANY_COLORS["Anthropic"],
    "google": COMPANY_COLORS["Google"],
    "meta-llama": COMPANY_COLORS["Meta"],
    "mistralai": COMPANY_COLORS["Mistral"],
    "deepseek": COMPANY_COLORS["DeepSeek"],
    "x-ai": COMPANY_COLORS["xAI"],
    "qwen": COMPANY_COLORS["Alibaba"],
    "Other": COMPANY_COLORS["Other"],
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
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})
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
            fixedrange=True,
        ),
        yaxis=dict(
            title=dict(text="Completion price ($ per 1M tokens)",
                       font=dict(size=11, color="#9CA3AF")),
            tickfont=dict(size=10, color="#6B7280"),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            type="log",
            fixedrange=True,
        ),
        dragmode=False,
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

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})

    # Insight: find the cheapest model overall (lowest prompt + completion)
    df["total_1m"] = df["prompt_1m"] + df["completion_1m"]
    cheapest = df.loc[df["total_1m"].idxmin()]
    total_models = len(df)

    insight = (
        f"{total_models} models compared. "
        f"Cheapest overall: {cheapest['name']} "
        f"(${cheapest['prompt_1m']:.2f} prompt + "
        f"${cheapest['completion_1m']:.2f} completion per 1M tokens)."
    )

    updated_text = _format_updated("OpenRouter API", "openrouter")

    return graph, insight, updated_text


# ── Callback 3: Downloads by org ───────────────────────────────────
DOWNLOAD_COLORS = {
    "Qwen (Alibaba)": COMPANY_COLORS["Alibaba"],
    "Meta": COMPANY_COLORS["Meta"],
    "OpenAI (community)": COMPANY_COLORS["OpenAI"],
    "OpenAI": COMPANY_COLORS["OpenAI"],
    "DeepSeek": COMPANY_COLORS["DeepSeek"],
    "NVIDIA": COMPANY_COLORS["NVIDIA"],
    "Microsoft": COMPANY_COLORS["Microsoft"],
    "Mistral": COMPANY_COLORS["Mistral"],
    "Google": COMPANY_COLORS["Google"],
    "Meta (legacy)": COMPANY_COLORS["Meta"],
    "EleutherAI": COMPANY_COLORS["Other"],
    "HuggingFace": COMPANY_COLORS["Other"],
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
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})
        return graph, "Data unavailable.", ""

    # Reverse for horizontal bar chart (highest at top)
    df = df.sort_values("downloads", ascending=True)

    # Format downloads for display (e.g., 138M, 31M)
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
            text=[_fmt_number(d) for d in df["downloads"]],
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
            range=[0, df["downloads"].max() * 1.25],
            fixedrange=True,
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#D1D5DB"),
            fixedrange=True,
        ),
        dragmode=False,
        margin=dict(l=120, r=50, t=10, b=10),
        height=400,
        autosize=True,
        bargap=0.25,
    )

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})

    # Insight
    top_org = df.iloc[-1]  # last row is highest (we sorted ascending)
    total_downloads = df["downloads"].sum()

    insight = (
        f"{top_org['display_name']} leads with {_fmt_number(top_org['downloads'])} downloads "
        f"across {top_org['model_count']} models in the top 200. "
        f"Total across top 12 orgs: {_fmt_number(total_downloads)}."
    )

    updated_text = _format_updated("HuggingFace API", "hf_downloads")

    return graph, insight, updated_text


# ── Callback 4: Model release timeline ─────────────────────────────
TIMELINE_COLORS = {k: v for k, v in COMPANY_COLORS.items() if k != "Other"}


@app.callback(
    Output("timeline-container", "children"),
    Output("timeline-insight", "children"),
    Output("timeline-updated", "children"),
    Input("refresh-24h", "n_intervals"),
)
def update_timeline(_n):
    try:
        raw_df = get_notable_models()
        benchmarks_models = get_benchmarks_models()
        df = build_timeline_data(raw_df, benchmarks_models=benchmarks_models)
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text=f"Could not load timeline: {e}",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#EF4444"))],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})
        return graph, "Data unavailable.", ""

    import numpy as np

    fig = go.Figure()

    df_with_compute = df[df["has_compute"]].copy()
    df_no_compute = filter_flagship_models(df[~df["has_compute"]])

    # ── Scatter: models WITH compute data ──
    for org in TIMELINE_COLORS:
        subset = df_with_compute[df_with_compute["org"] == org]
        if subset.empty:
            continue

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
            legendgroup=org,
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

    # ── Rug plot: models WITHOUT compute data (e.g. 2026 models) ──
    # Place them at the bottom of the log Y axis as diamond markers
    if not df_no_compute.empty and not df_with_compute.empty:
        # Position just below the minimum compute value on log scale
        y_min = df_with_compute["compute_flop"].min()
        rug_y = y_min * 0.15  # slightly below the lowest compute point

        for org in TIMELINE_COLORS:
            subset = df_no_compute[df_no_compute["org"] == org]
            if subset.empty:
                continue

            # Only show legend entry if this org doesn't already have a compute trace
            has_compute_trace = not df_with_compute[df_with_compute["org"] == org].empty

            fig.add_trace(go.Scatter(
                x=subset["date"],
                y=[rug_y] * len(subset),
                mode="markers+text",
                name=org,
                legendgroup=org,
                showlegend=not has_compute_trace,
                marker=dict(
                    color=TIMELINE_COLORS.get(org, "#6B7280"),
                    size=10,
                    symbol="diamond",
                    opacity=0.9,
                    line=dict(width=1, color="#D1D5DB"),
                ),
                text=subset["model"],
                textposition="top center",
                textfont=dict(size=8, color="#9CA3AF",
                              family="IBM Plex Mono, monospace"),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}<br>"
                    "Date: %{x|%b %Y}<br>"
                    "Compute: not yet published"
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
            fixedrange=True,
        ),
        yaxis=dict(
            title=dict(text="Training compute (FLOP)",
                       font=dict(size=11, color="#9CA3AF")),
            tickfont=dict(size=10, color="#6B7280"),
            gridcolor="rgba(255,255,255,0.06)",
            type="log",
            fixedrange=True,
        ),
        dragmode=False,
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

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})

    # Insight
    org_counts = df["org"].value_counts()
    most_active = org_counts.index[0]
    most_active_count = org_counts.iloc[0]
    num_recent = len(df_no_compute)

    if not df_with_compute.empty:
        biggest = df_with_compute.loc[df_with_compute["compute_flop"].idxmax()]
        insight = (
            f"Since 2020, {most_active} has released {most_active_count} notable models "
            f"— more than anyone else. The largest training run recorded: "
            f"{biggest['model']} by {biggest['org']} "
            f"({biggest['compute_flop']:.1e} FLOP)."
        )
        if num_recent > 0:
            insight += (
                f" {num_recent} recent models are awaiting compute data."
            )
    else:
        insight = f"{most_active} leads with {most_active_count} notable models since 2020."

    updated_text = _format_updated("Epoch AI", "notable")

    return graph, insight, updated_text


# ── Callback 6: Context window distribution ───────────────────────

CONTEXT_BUCKET_COLORS = {
    "≤ 8K": "#1B2B4B",
    "8–32K": "#374151",
    "32–128K": "#6B7280",
    "128–256K": "#9CA3AF",
    "256K–1M": "#D1D5DB",
    "> 1M": "#F3F4F6",
}


@app.callback(
    Output("context-container", "children"),
    Output("context-insight", "children"),
    Output("context-updated", "children"),
    Input("refresh-5min", "n_intervals"),
)
def update_context(_n):
    try:
        raw_models = get_openrouter_models()
        bucket_df, stats = build_context_data(raw_models)
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text=f"Could not load context data: {e}",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#EF4444"))],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})
        return graph, "Data unavailable.", ""

    bar_colors = [CONTEXT_BUCKET_COLORS.get(b, "#374151") for b in bucket_df["bucket"]]

    fig = go.Figure(
        data=go.Bar(
            x=bucket_df["count"],
            y=bucket_df["bucket"],
            orientation="h",
            marker=dict(
                color=bar_colors,
                cornerradius=4,
            ),
            text=bucket_df["count"],
            textposition="outside",
            textfont=dict(size=12, color="#D1D5DB", family="IBM Plex Mono, monospace"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Models: %{x}<br>"
                "e.g. %{customdata}"
                "<extra></extra>"
            ),
            customdata=bucket_df["examples"],
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D1D5DB", family="Outfit, system-ui, sans-serif"),
        xaxis=dict(
            visible=False,
            range=[0, bucket_df["count"].max() * 1.2],
            fixedrange=True,
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#D1D5DB"),
            categoryorder="array",
            categoryarray=list(reversed(bucket_df["bucket"].tolist())),
            fixedrange=True,
        ),
        dragmode=False,
        margin=dict(l=90, r=50, t=10, b=10),
        height=300,
        autosize=True,
        bargap=0.3,
    )

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})

    # Insight
    total = stats["total_unique"]
    max_m = stats["max_model"]
    big_bucket = bucket_df[bucket_df["bucket"] == "> 1M"]["count"].iloc[0]
    sweet_spot = bucket_df[bucket_df["bucket"] == "32–128K"]["count"].iloc[0]

    def fmt_ctx(tokens):
        if tokens >= 1_000_000:
            return f"{tokens / 1_000_000:.1f}M"
        return f"{tokens // 1_000}K"

    insight = (
        f"{total} unique model families. "
        f"The sweet spot is 32–128K tokens ({sweet_spot} models), "
        f"but {big_bucket} models now exceed 1M tokens — "
        f"led by {max_m['name']} at {fmt_ctx(max_m['ctx'])}."
    )

    updated_text = _format_updated("OpenRouter API", "openrouter")

    return graph, insight, updated_text


# ── Callback 7: AI chip dominance ─────────────────────────────────

CHIP_COUNTRY_COLORS = {
    "US": "#60A5FA",     # blue
    "China": "#EF4444",  # red
    "Other": "#6B7280",  # gray
}


@app.callback(
    Output("chips-container", "children"),
    Output("chips-insight", "children"),
    Output("chips-updated", "children"),
    Input("refresh-24h", "n_intervals"),
)
def update_chips(_n):
    try:
        raw_df = get_ml_hardware()
        chip_df, stats = build_chip_dominance_data(raw_df, top_n=10)
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text=f"Could not load chip data: {e}",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#EF4444"))],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})
        return graph, "Data unavailable.", ""

    # Reverse for horizontal bar (highest at top)
    chip_df = chip_df.sort_values("count", ascending=True)

    bar_colors = [CHIP_COUNTRY_COLORS.get(c, "#6B7280") for c in chip_df["country"]]

    fig = go.Figure(
        data=go.Bar(
            x=chip_df["count"],
            y=chip_df["manufacturer"],
            orientation="h",
            marker=dict(
                color=bar_colors,
                cornerradius=4,
            ),
            text=chip_df["count"],
            textposition="outside",
            textfont=dict(size=12, color="#D1D5DB", family="IBM Plex Mono, monospace"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "AI chips: %{x}<br>"
                "Origin: %{customdata}"
                "<extra></extra>"
            ),
            customdata=chip_df["country"],
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D1D5DB", family="Outfit, system-ui, sans-serif"),
        xaxis=dict(
            visible=False,
            range=[0, chip_df["count"].max() * 1.2],
            fixedrange=True,
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#D1D5DB"),
            fixedrange=True,
        ),
        dragmode=False,
        margin=dict(l=160, r=50, t=10, b=10),
        height=380,
        autosize=True,
        bargap=0.25,
    )

    # Add legend manually as annotations (since bar chart doesn't support grouped legend well)
    for i, (country, color) in enumerate(CHIP_COUNTRY_COLORS.items()):
        fig.add_annotation(
            x=1.0, y=1.0 - i * 0.08,
            xref="paper", yref="paper",
            text=f"● {country}",
            showarrow=False,
            font=dict(size=11, color=color, family="Outfit, system-ui, sans-serif"),
            xanchor="right",
        )

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})

    # Insight
    insight = (
        f"{stats['top_maker']} dominates with {stats['top_count']} of "
        f"{stats['total_chips']} registered AI chips ({stats['top_share']:.0f}%). "
        f"The hidden story: China has {stats['cn_makers']} chip manufacturers "
        f"({stats['cn_chips']} chips) vs {stats['us_makers']} for the US "
        f"({stats['us_chips']} chips)."
    )

    updated_text = _format_updated("Epoch AI ML hardware", "ml_hardware")

    return graph, insight, updated_text


# ── Callback 8: GPU cluster world map ─────────────────────────────
GPU_REGION_COLORS = {
    "China": "#EF4444",          # red
    "United States": "#60A5FA",  # blue
    "Europe": "#2DD4BF",         # teal
    "Other": "#6B7280",          # gray
}

GPU_REGION_SYMBOLS = {
    "China": "diamond",
    "United States": "circle",
    "Europe": "triangle-up",
    "Other": "square",
}


@app.callback(
    Output("gpu-map-container", "children"),
    Output("gpu-map-insight", "children"),
    Output("gpu-map-updated", "children"),
    Input("refresh-24h", "n_intervals"),
)
def update_gpu_map(_n):
    try:
        raw_df = get_gpu_clusters()
        map_df, stats = build_gpu_map_data(raw_df)
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text=f"Could not load GPU clusters: {e}",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#EF4444"))],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})
        return graph, "Data unavailable.", ""

    fig = go.Figure()

    def fmt_power(mw):
        if pd.notna(mw):
            return f"{mw:,.0f} MW"
        return "Unknown"

    def fmt_h100(h):
        if h >= 1_000_000:
            return f"{h / 1_000_000:,.1f}M"
        if h >= 1_000:
            return f"{h / 1_000:,.0f}K"
        return f"{h:,.0f}"

    # One trace per region for legend + color
    # Anonymized clusters get a distinct hollow diamond marker
    for region, color in GPU_REGION_COLORS.items():
        subset = map_df[(map_df["region"] == region) & (~map_df["anonymized"])]
        if not subset.empty:
            fig.add_trace(go.Scattergeo(
                lat=subset["lat"],
                lon=subset["lon"],
                mode="markers",
                name=f"{region} ({len(subset)})",
                marker=dict(
                    color=color,
                    size=subset["dot_size"],
                    opacity=0.7,
                    symbol=GPU_REGION_SYMBOLS.get(region, "circle"),
                    line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
                    sizemode="diameter",
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Owner: %{customdata[1]}<br>"
                    "Country: %{customdata[2]}<br>"
                    "Chip: %{customdata[3]}<br>"
                    "H100 equiv: %{customdata[4]}<br>"
                    "Power: %{customdata[5]}<br>"
                    "Status: %{customdata[6]}"
                    "<extra></extra>"
                ),
                customdata=list(zip(
                    subset["name"],
                    subset["owner"],
                    subset["country"],
                    subset["chip_type"],
                    [fmt_h100(h) for h in subset["h100_equiv"]],
                    [fmt_power(mw) for mw in subset["power_mw"]],
                    subset["status"],
                )),
            ))

        # Anonymized clusters — hollow markers with lower opacity
        anon_subset = map_df[(map_df["region"] == region) & (map_df["anonymized"])]
        if not anon_subset.empty:
            fig.add_trace(go.Scattergeo(
                lat=anon_subset["lat"],
                lon=anon_subset["lon"],
                mode="markers",
                name=f"{region} — approx. ({len(anon_subset)})",
                marker=dict(
                    color="rgba(0,0,0,0)",
                    size=anon_subset["dot_size"],
                    opacity=0.5,
                    symbol="diamond",
                    line=dict(width=1.5, color=color),
                    sizemode="diameter",
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Owner: %{customdata[1]}<br>"
                    "Country: %{customdata[2]}<br>"
                    "Chip: %{customdata[3]}<br>"
                    "H100 equiv: %{customdata[4]}<br>"
                    "Power: %{customdata[5]}<br>"
                    "Status: %{customdata[6]}<br>"
                    "<i>Location approximate</i>"
                    "<extra></extra>"
                ),
                customdata=list(zip(
                    anon_subset["name"],
                    anon_subset["owner"],
                    anon_subset["country"],
                    anon_subset["chip_type"],
                    [fmt_h100(h) for h in anon_subset["h100_equiv"]],
                    [fmt_power(mw) for mw in anon_subset["power_mw"]],
                    anon_subset["status"],
                )),
            ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D1D5DB", family="Outfit, system-ui, sans-serif"),
        geo=dict(
            bgcolor="rgba(0,0,0,0)",
            showland=True,
            landcolor="#1F2937",
            showocean=True,
            oceancolor="#111318",
            showcountries=True,
            countrycolor="#374151",
            showcoastlines=True,
            coastlinecolor="#374151",
            showlakes=False,
            showframe=False,
            projection_type="natural earth",
        ),
        legend=dict(
            font=dict(size=11, color="#D1D5DB"),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
        ),
        dragmode=False,
        margin=dict(l=0, r=0, t=40, b=10),
        height=500,
        autosize=True,
    )

    # Annotation for anonymized China clusters
    n_anon = map_df["anonymized"].sum()
    if n_anon > 0:
        fig.add_annotation(
            x=0.82, y=0.35,
            xref="paper", yref="paper",
            text=f"{n_anon} clusters<br><i>(locations anonymized)</i>",
            showarrow=False,
            font=dict(size=10, color="#EF4444", family="Outfit, system-ui, sans-serif"),
            bgcolor="rgba(17,19,24,0.75)",
            bordercolor="#EF4444",
            borderwidth=1,
            borderpad=6,
        )

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})

    # Insight: geopolitical + who dominates
    cc = stats["country_counts"]
    us_count = cc.get("United States of America", 0)
    cn_count = cc.get("China", 0)
    total = stats["total_clusters"]
    total_mw = stats["total_mw"]

    # Top owners
    top_owners = stats["top_owners"]
    owner_parts = []
    for owner, h100 in top_owners.head(3).items():
        if h100 >= 1_000_000:
            owner_parts.append(f"{owner} ({h100 / 1_000_000:,.1f}M)")
        else:
            owner_parts.append(f"{owner} ({h100 / 1_000:,.0f}K)")

    def fmt_mw(mw):
        if pd.notna(mw):
            return f"{mw:,.0f}"
        return "?"

    insight = (
        f"{total} GPU clusters across {len(cc)} countries. "
        f"China leads with {cn_count} clusters vs {us_count} for the US. "
        f"Total power: {fmt_mw(total_mw)} MW. "
        f"Biggest investors by H100 equivalents: {', '.join(owner_parts)}."
    )

    updated_text = _format_updated("Epoch AI GPU clusters", "gpu_clusters")

    return graph, insight, updated_text


# ── Callback 9: Trending models ──────────────────────────────────
HF_ORG_TO_CANONICAL = {
    "Qwen": "Alibaba", "meta-llama": "Meta", "openai": "OpenAI",
    "deepseek-ai": "DeepSeek", "google": "Google", "microsoft": "Microsoft",
    "mistralai": "Mistral", "nvidia": "NVIDIA", "x-ai": "xAI",
    "ByteDance": "ByteDance", "baidu": "Baidu",
}


@app.callback(
    Output("trending-container", "children"),
    Output("trending-insight", "children"),
    Output("trending-updated", "children"),
    Input("refresh-5min", "n_intervals"),
)
def update_trending(_n):
    try:
        raw = get_hf_trending()
        df = build_trending_data(raw)
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text=f"Could not load trending: {e}",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#EF4444"))],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})
        return graph, "Data unavailable.", ""

    df = df.sort_values("downloads", ascending=True)

    bar_colors = [
        COMPANY_COLORS.get(HF_ORG_TO_CANONICAL.get(org, ""), COMPANY_COLORS["Other"])
        for org in df["org"]
    ]

    fig = go.Figure(data=go.Bar(
        x=df["downloads"],
        y=[f"#{r}  {n}" for r, n in zip(df["rank"], df["model_name"])],
        orientation="h",
        marker=dict(color=bar_colors, cornerradius=4),
        text=[_fmt_number(d) for d in df["downloads"]],
        textposition="outside",
        textfont=dict(size=10, color="#D1D5DB", family="IBM Plex Mono, monospace"),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Org: %{customdata[1]}<br>"
            "Downloads: %{x:,.0f}<br>"
            "Likes: %{customdata[2]:,.0f}<br>"
            "Age: %{customdata[3]} days"
            "<extra></extra>"
        ),
        customdata=list(zip(df["model_name"], df["org"], df["likes"], df["days_ago"])),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D1D5DB", family="Outfit, system-ui, sans-serif"),
        xaxis=dict(visible=False, range=[0, df["downloads"].max() * 1.25], fixedrange=True),
        yaxis=dict(tickfont=dict(size=10, color="#D1D5DB"), fixedrange=True),
        dragmode=False,
        margin=dict(l=180, r=50, t=10, b=10),
        height=480,
        autosize=True,
        bargap=0.2,
    )

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})

    top = df.iloc[-1]
    newest = df[df["days_ago"].notna()].nsmallest(1, "days_ago")
    insight = f"#{int(top['rank'])} {top['model_name']} by {top['org']} leads with {_fmt_number(top['downloads'])} downloads."
    if not newest.empty:
        n = newest.iloc[0]
        insight += f" Freshest: {n['model_name']} ({int(n['days_ago'])} days old)."

    updated_text = _format_updated("HuggingFace Trending", "hf_trending")
    return graph, insight, updated_text


# ── Callback 10: Modality champions ──────────────────────────────
@app.callback(
    Output("modality-container", "children"),
    Output("modality-insight", "children"),
    Output("modality-updated", "children"),
    Input("refresh-5min", "n_intervals"),
)
def update_modality(_n):
    try:
        raw = get_hf_modality_champions()
        df = build_modality_champions(raw)
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text=f"Could not load modality data: {e}",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#EF4444"))],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})
        return graph, "Data unavailable.", ""

    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text="No modality data available",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#9CA3AF"))],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})
        return graph, "No modality data available.", ""

    df = df.sort_values("downloads", ascending=True)

    fig = go.Figure(data=go.Bar(
        x=df["downloads"],
        y=df["label"],
        orientation="h",
        marker=dict(color="#6B7280", cornerradius=4),
        text=[f"{name} ({org})" for name, org in zip(df["model_name"], df["display_org"])],
        textposition="outside",
        textfont=dict(size=10, color="#9CA3AF", family="IBM Plex Mono, monospace"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Model: %{customdata[0]}<br>"
            "Org: %{customdata[1]}<br>"
            "Downloads: %{x:,.0f}<br>"
            "Likes: %{customdata[2]:,.0f}"
            "<extra></extra>"
        ),
        customdata=list(zip(df["model_name"], df["display_org"], df["likes"])),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D1D5DB", family="Outfit, system-ui, sans-serif"),
        xaxis=dict(visible=False, range=[0, df["downloads"].max() * 1.4], fixedrange=True),
        yaxis=dict(tickfont=dict(size=12, color="#D1D5DB"), fixedrange=True),
        dragmode=False,
        margin=dict(l=110, r=160, t=10, b=10),
        height=320,
        autosize=True,
        bargap=0.3,
    )

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})

    top = df.iloc[-1]
    insight = (
        f"Across 7 AI modalities, {top['display_org']}'s {top['model_name']} "
        f"leads {top['label']} with {_fmt_number(top['downloads'])} downloads."
    )

    updated_text = _format_updated("HuggingFace Modality", "hf_modality")
    return graph, insight, updated_text


# ── Callback 11: SWE-bench leaderboard ──────────────────────────────
@app.callback(
    Output("swebench-container", "children"),
    Output("swebench-insight", "children"),
    Output("swebench-updated", "children"),
    Input("refresh-24h", "n_intervals"),
)
def update_swebench(_n):
    try:
        raw = get_swebench_leaderboard()
        df = build_swebench_data(raw, top_n=15)
    except Exception as e:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            annotations=[dict(text=f"Could not load SWE-bench: {e}",
                              xref="paper", yref="paper", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=14, color="#EF4444"))],
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        graph = dcc.Graph(figure=empty_fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})
        return graph, "Data unavailable.", ""

    df = df.sort_values("resolved", ascending=True)

    bar_colors = [
        COMPANY_COLORS.get(org, COMPANY_COLORS["Other"]) for org in df["org"]
    ]

    fig = go.Figure(data=go.Bar(
        x=df["resolved"],
        y=df["name"],
        orientation="h",
        marker=dict(color=bar_colors, cornerradius=4),
        text=[f"{v:.1f}%" for v in df["resolved"]],
        textposition="outside",
        textfont=dict(size=10, color="#D1D5DB", family="IBM Plex Mono, monospace"),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Resolved: %{x:.1f}%<br>"
            "Org: %{customdata[0]}<br>"
            "Date: %{customdata[1]}<br>"
            "Open source: %{customdata[2]}"
            "<extra></extra>"
        ),
        customdata=list(zip(
            df["org"],
            df["date"],
            ["Yes" if o else "No" for o in df["is_open"]],
        )),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#D1D5DB", family="Outfit, system-ui, sans-serif"),
        xaxis=dict(
            title=dict(text="Issues resolved (%)", font=dict(size=11, color="#9CA3AF")),
            tickfont=dict(size=10, color="#6B7280"),
            range=[0, min(df["resolved"].max() * 1.15, 100)],
            fixedrange=True,
            gridcolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(tickfont=dict(size=10, color="#D1D5DB"), fixedrange=True),
        dragmode=False,
        margin=dict(l=240, r=50, t=10, b=35),
        height=480,
        autosize=True,
        bargap=0.2,
    )

    graph = dcc.Graph(figure=fig, config={"displayModeBar": False, "responsive": True, "scrollZoom": False})

    # Insight
    top = df.iloc[-1]
    open_count = df["is_open"].sum()
    insight = (
        f"{top['name']} leads at {top['resolved']:.1f}% of real GitHub issues resolved. "
        f"{open_count} of the top {len(df)} systems use open-source components."
    )

    updated_text = _format_updated("SWE-bench", "swebench")
    return graph, insight, updated_text


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
