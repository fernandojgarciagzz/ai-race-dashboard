# The AI Race Dashboard

A live, data-driven dashboard tracking the global AI competition across benchmarks, pricing, compute, and more. Built with Plotly Dash.

**Live at**: [ai-race-dashboard.onrender.com](https://ai-race-dashboard.onrender.com)

## Charts

| # | Title | Source | Refresh |
|---|-------|--------|---------|
| 1 | **Which AI model is the best right now?** — Benchmark heatmap across Math, Code, Reasoning, Science, Common Sense | Epoch AI `eci_benchmarks.csv` | 24h |
| 2 | **How much memory can AI hold?** — Context window distribution (≤8K to >1M tokens) | OpenRouter API | 5min |
| 3 | **How much does AI cost and who's cheapest?** — Pricing scatter (prompt vs completion per 1M tokens) | OpenRouter API | 5min |
| 4 | **Who's winning the open-source AI race?** — Downloads by organization | HuggingFace API | 5min |
| 5 | **Who's investing most in winning this race?** — Model release timeline with training compute | Epoch AI `notable_ai_models.csv` + `eci_benchmarks.csv` | 24h |
| 6 | **What can each company's AI actually do?** — Multimodal capabilities heatmap | OpenRouter API | 5min |
| 7 | **Who makes the chips powering AI?** — Chip manufacturer dominance (NVIDIA 52%, US vs China) | Epoch AI `ml_hardware.csv` | 24h |
| 8 | **Where is AI compute? A global map** — 691 GPU clusters plotted on a world map | Epoch AI `gpu_clusters.csv` | 24h |

## Data Sources

- **[Epoch AI](https://epoch.ai)** (CC BY 4.0) — Benchmarks, notable models, GPU clusters
- **[OpenRouter](https://openrouter.ai)** — Model pricing, context lengths, capabilities
- **[HuggingFace](https://huggingface.co)** — Download counts, trending models

All data is fetched live at runtime. No static datasets are stored in the repo.

## Architecture

```
app.py              Dash layout + callbacks (one per chart)
data/
  fetch.py          Downloads & caches data from external APIs
  process.py        Transforms raw data into chart-ready DataFrames
assets/
  style.css         Dark theme styling
```

**Caching strategy**:
- Epoch CSVs: cached 24 hours in memory, stale fallback on failure
- OpenRouter / HuggingFace: cached 5 minutes

## Run locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:8050
```

## Deploy

Configured for [Render](https://render.com) via `render.yaml`. Pushes to `main` trigger automatic deploys.

```yaml
startCommand: gunicorn app:server --bind 0.0.0.0:$PORT
```

## Notable design decisions

- **No API keys required** — All data sources are public APIs or CC-licensed CSVs.
- **No static datasets** — Everything is fetched live, so the dashboard stays current without manual updates.
- **Timeline 2026 gap** — `notable_ai_models.csv` ends in Dec 2025. Recent models (GPT-5.4 Pro, Claude Opus 4.6, Gemini 3.1 Pro, etc.) are supplemented from `eci_benchmarks.csv` and displayed as diamond markers with "compute not yet published".
- **China GPU clusters** — Epoch AI anonymizes Chinese cluster locations to a single centroid. The map applies jitter to visualize all 253 clusters with an annotation noting the anonymization.
- **Context window dedup** — OpenRouter lists multiple variants of the same model. The context chart deduplicates by family and keeps the variant with the largest context window.
