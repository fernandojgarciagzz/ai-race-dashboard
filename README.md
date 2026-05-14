# The AI Race Dashboard

A live, data-driven dashboard tracking the global AI competition across benchmarks, pricing, compute, and more. Built with Plotly Dash.

**Live at**: [ai-race-dashboard.onrender.com](https://ai-race-dashboard.onrender.com)

## Charts

| # | Title | Source | Refresh |
|---|-------|--------|---------|
| 1 | **Which AI model is the best right now?** — Benchmark heatmap across Math, Code, Reasoning, Science, Common Sense (z-score normalized) | Epoch AI `eci_benchmarks.csv` | 24h |
| 2 | **How much memory can AI hold?** — Context window distribution (≤8K to >1M tokens) | OpenRouter API | 5min |
| 3 | **How much does AI cost and who's cheapest?** — Pricing scatter (prompt vs completion per 1M tokens) | OpenRouter API | 5min |
| 4 | **Who's investing most in winning this race?** — Model release timeline with training compute | Epoch AI `notable_ai_models.csv` + `eci_benchmarks.csv` | 24h |
| 5 | **What can each company's AI actually do?** — Multimodal capabilities heatmap (deduplicated by family) | OpenRouter API | 5min |
| 6 | **Who makes the chips powering AI?** — Chip manufacturer dominance (US vs China) | Epoch AI `ml_hardware.csv` | 24h |
| 7 | **Where is AI compute? A global map** — GPU clusters plotted on a world map (anonymized clusters shown as hollow diamonds) | Epoch AI `gpu_clusters.csv` | 24h |
| 8 | **Who's winning the open-source AI race?** — Downloads by org, recency-weighted (60-day half-life) | HuggingFace API | 5min |
| 9 | **What's trending in AI right now?** — Top 15 trending text-generation models | HuggingFace API | 5min |
| 10 | **Who leads each AI modality?** — #1 model per capability (images, voice, video, etc.) | HuggingFace API | 5min |
| 11 | **Which AI can actually code?** — SWE-bench real-world coding leaderboard | SWE-bench | 6h |
| 12 | **Which AI SDK are developers choosing?** — Weekly PyPI installs for major AI packages | PyPI Stats | 24h |

## Data Sources

- **[Epoch AI](https://epoch.ai)** (CC BY 4.0) — Benchmarks, notable models, GPU clusters, ML hardware
- **[OpenRouter](https://openrouter.ai)** — Model pricing, context lengths, capabilities
- **[HuggingFace](https://huggingface.co)** — Download counts, trending, modality leaders
- **[SWE-bench](https://www.swebench.com)** — Real-world software engineering benchmark
- **[PyPI Stats](https://pypistats.org)** — Python package download statistics

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

- **Zero maintenance** — All data is fetched live from public APIs. No static datasets, no API keys, no manual updates.
- **Two-layer explanations** — Each chart has a static description (how to read it) and a dynamic insight (what the data says), both generated from live data.
- **Mobile-friendly** — All charts have zoom/drag/pinch disabled (`fixedrange`, `dragmode=False`, `scrollZoom=False`) to prevent accidental zoom on touch devices.
- **Timeline 2026 gap** — `notable_ai_models.csv` ends in Dec 2025. Recent models (GPT-5.4 Pro, Claude Opus 4.6, Gemini 3.1 Pro, etc.) are supplemented from `eci_benchmarks.csv` and displayed as diamond markers. A flagship filter deduplicates model families and keeps only the last 6 months.
- **China GPU clusters** — Epoch AI anonymizes Chinese cluster locations to a single centroid. The map applies jitter to visualize all 253 clusters with an annotation noting the anonymization.
- **Context window dedup** — OpenRouter lists multiple variants of the same model. The context chart deduplicates by family and keeps the variant with the largest context window.
- **Chip geopolitics** — Manufacturer country of origin is mapped manually (`CHIP_MANUFACTURER_COUNTRY` in process.py) to color bars by US/China/Other.
