"""
fetch.py — Downloads data from all sources and caches in memory.

Sources:
- Epoch AI (CSV, 24h cache): benchmarks
- OpenRouter (JSON, live per visit): model pricing

Cache strategy:
- Epoch CSVs: cached 24 hours, stale fallback on failure.
- OpenRouter: cached 5 minutes (prices change rarely, avoids hammering API).
"""

import pandas as pd
import time
import io
import requests

# ── Module-level caches ─────────────────────────────────────────────
_benchmarks_cache = None
_benchmarks_cache_time = 0

_notable_cache = None
_notable_cache_time = 0

_openrouter_cache = None
_openrouter_cache_time = 0

_hf_downloads_cache = None
_hf_downloads_cache_time = 0

_hf_trending_cache = None
_hf_trending_cache_time = 0

_hf_modality_cache = None
_hf_modality_cache_time = 0

_gpu_clusters_cache = None
_gpu_clusters_cache_time = 0

_ml_hardware_cache = None
_ml_hardware_cache_time = 0

_swebench_cache = None
_swebench_cache_time = 0

_pypi_cache = None
_pypi_cache_time = 0

_lmarena_cache = None
_lmarena_cache_time = 0

CACHE_24H = 86400
CACHE_6H = 21600
CACHE_5MIN = 300

BENCHMARKS_URL = "https://epoch.ai/data/eci_benchmarks.csv"
NOTABLE_URL = "https://epoch.ai/data/notable_ai_models.csv"
GPU_CLUSTERS_URL = "https://epoch.ai/data/gpu_clusters.csv"
ML_HARDWARE_URL = "https://epoch.ai/data/ml_hardware.csv"
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
HF_DOWNLOADS_URL = "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=200&pipeline_tag=text-generation"
HF_TRENDING_URL = "https://huggingface.co/api/models?sort=trendingScore&direction=-1&limit=15&pipeline_tag=text-generation"
SWEBENCH_URL = "https://www.swebench.com/"
LMARENA_LEADERBOARD_URL = "https://huggingface.co/spaces/lmarena-ai/arena-leaderboard/tree/main"
PYPI_PACKAGES = ["openai", "anthropic", "transformers", "google-generativeai",
                 "langchain", "together", "mistralai", "cohere", "replicate"]


def get_last_fetch_time(source: str) -> float:
    """Returns the Unix timestamp of the last successful fetch for a given source."""
    cache_times = {
        "benchmarks": _benchmarks_cache_time,
        "notable": _notable_cache_time,
        "openrouter": _openrouter_cache_time,
        "hf_downloads": _hf_downloads_cache_time,
        "hf_trending": _hf_trending_cache_time,
        "hf_modality": _hf_modality_cache_time,
        "gpu_clusters": _gpu_clusters_cache_time,
        "ml_hardware": _ml_hardware_cache_time,
        "swebench": _swebench_cache_time,
        "pypi": _pypi_cache_time,
        "lmarena": _lmarena_cache_time,
    }
    return cache_times.get(source, 0)


def get_benchmarks():
    """
    Returns the Epoch AI benchmarks DataFrame.

    How it works:
    1. Checks if we have cached data that's less than 24 hours old.
    2. If yes, returns the cached DataFrame (fast, no network call).
    3. If no, downloads fresh data from Epoch AI, caches it, returns it.
    4. If the download fails, returns the stale cache (if any) or raises an error.

    Returns:
        pd.DataFrame with columns: model, model_group, benchmark, performance, date
    """
    global _benchmarks_cache, _benchmarks_cache_time

    now = time.time()
    cache_is_fresh = (now - _benchmarks_cache_time) < CACHE_24H

    if _benchmarks_cache is not None and cache_is_fresh:
        return _benchmarks_cache

    try:
        # requests.get instead of pd.read_csv(url) for better error handling
        response = requests.get(BENCHMARKS_URL, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))
        _benchmarks_cache = df
        _benchmarks_cache_time = now
        return df

    except Exception as e:
        if _benchmarks_cache is not None:
            print(f"Warning: Failed to refresh benchmarks ({e}). Using stale cache.")
            return _benchmarks_cache
        raise RuntimeError(
            f"Could not fetch benchmarks data and no cache available: {e}"
        )


def get_notable_models():
    """
    Returns the Epoch AI notable models DataFrame.
    Cached for 24 hours (same as benchmarks — batch data).
    """
    global _notable_cache, _notable_cache_time

    now = time.time()
    if _notable_cache is not None and (now - _notable_cache_time) < CACHE_24H:
        return _notable_cache

    try:
        response = requests.get(NOTABLE_URL, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        _notable_cache = df
        _notable_cache_time = now
        return df

    except Exception as e:
        if _notable_cache is not None:
            print(f"Warning: Failed to refresh notable models ({e}). Using stale cache.")
            return _notable_cache
        raise RuntimeError(
            f"Could not fetch notable models and no cache available: {e}"
        )


def get_gpu_clusters():
    """
    Returns the Epoch AI GPU clusters DataFrame.
    Cached for 24 hours (batch data, updated infrequently).
    """
    global _gpu_clusters_cache, _gpu_clusters_cache_time

    now = time.time()
    if _gpu_clusters_cache is not None and (now - _gpu_clusters_cache_time) < CACHE_24H:
        return _gpu_clusters_cache

    try:
        response = requests.get(GPU_CLUSTERS_URL, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        _gpu_clusters_cache = df
        _gpu_clusters_cache_time = now
        return df

    except Exception as e:
        if _gpu_clusters_cache is not None:
            print(f"Warning: Failed to refresh GPU clusters ({e}). Using stale cache.")
            return _gpu_clusters_cache
        raise RuntimeError(
            f"Could not fetch GPU clusters and no cache available: {e}"
        )


def get_ml_hardware():
    """
    Returns the Epoch AI ML hardware (AI chips) DataFrame.
    Cached for 24 hours.
    """
    global _ml_hardware_cache, _ml_hardware_cache_time

    now = time.time()
    if _ml_hardware_cache is not None and (now - _ml_hardware_cache_time) < CACHE_24H:
        return _ml_hardware_cache

    try:
        response = requests.get(ML_HARDWARE_URL, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        _ml_hardware_cache = df
        _ml_hardware_cache_time = now
        return df

    except Exception as e:
        if _ml_hardware_cache is not None:
            print(f"Warning: Failed to refresh ML hardware ({e}). Using stale cache.")
            return _ml_hardware_cache
        raise RuntimeError(
            f"Could not fetch ML hardware and no cache available: {e}"
        )


def get_openrouter_models():
    """
    Returns OpenRouter model list as a list of dicts (raw JSON).

    Cached for 5 minutes — prices change rarely but we don't want
    to hit the API on every single page load.

    Each model dict has: id, name, pricing.prompt, pricing.completion,
    context_length, architecture, created.
    """
    global _openrouter_cache, _openrouter_cache_time

    now = time.time()
    if _openrouter_cache is not None and (now - _openrouter_cache_time) < CACHE_5MIN:
        return _openrouter_cache

    try:
        response = requests.get(OPENROUTER_URL, timeout=15)
        response.raise_for_status()
        models = response.json()["data"]
        _openrouter_cache = models
        _openrouter_cache_time = now
        return models

    except Exception as e:
        if _openrouter_cache is not None:
            print(f"Warning: Failed to refresh OpenRouter ({e}). Using stale cache.")
            return _openrouter_cache
        raise RuntimeError(
            f"Could not fetch OpenRouter data and no cache available: {e}"
        )


def get_benchmarks_models():
    """
    Extracts unique models from the benchmarks CSV for the timeline chart.

    Returns models with: model_group (name), date, org (inferred from name).
    These supplement notable_ai_models.csv which lacks 2026 models.
    """
    df = get_benchmarks()

    # Infer org from model name
    ORG_PATTERNS = [
        (["GPT-", "o1", "o3", "o4"], "OpenAI"),
        (["Claude"], "Anthropic"),
        (["Gemini"], "Google"),
        (["Llama", "LLaMA"], "Meta"),
        (["DeepSeek"], "DeepSeek"),
        (["Qwen"], "Alibaba"),
        (["Kimi"], "Moonshot"),
        (["GLM"], "Zhipu AI"),
        (["MiniMax"], "MiniMax"),
        (["Mistral", "Magistral"], "Mistral"),
        (["Grok"], "xAI"),
    ]

    def infer_org(name):
        for prefixes, org in ORG_PATTERNS:
            for prefix in prefixes:
                if name.startswith(prefix):
                    return org
        return None

    # Get unique model_group + latest date
    grouped = (
        df.groupby("model_group")["date"]
        .max()
        .reset_index()
    )
    grouped["org"] = grouped["model_group"].apply(infer_org)
    # Drop models where org couldn't be inferred
    grouped = grouped[grouped["org"].notna()].reset_index(drop=True)
    grouped["date"] = pd.to_datetime(grouped["date"], errors="coerce")
    grouped = grouped.dropna(subset=["date"])

    return grouped


def get_hf_top_models():
    """
    Returns top 200 text-generation models from HuggingFace sorted by downloads.

    Cached for 5 minutes. Returns a list of dicts with: id, downloads, likes,
    pipeline_tag, library_name, createdAt.
    """
    global _hf_downloads_cache, _hf_downloads_cache_time

    now = time.time()
    if _hf_downloads_cache is not None and (now - _hf_downloads_cache_time) < CACHE_5MIN:
        return _hf_downloads_cache

    try:
        response = requests.get(HF_DOWNLOADS_URL, timeout=15)
        response.raise_for_status()
        models = response.json()
        _hf_downloads_cache = models
        _hf_downloads_cache_time = now
        return models

    except Exception as e:
        if _hf_downloads_cache is not None:
            print(f"Warning: Failed to refresh HuggingFace ({e}). Using stale cache.")
            return _hf_downloads_cache
        raise RuntimeError(
            f"Could not fetch HuggingFace data and no cache available: {e}"
        )


def get_hf_trending():
    """
    Returns top 15 trending text-generation models from HuggingFace.
    Cached for 5 minutes.
    """
    global _hf_trending_cache, _hf_trending_cache_time

    now = time.time()
    if _hf_trending_cache is not None and (now - _hf_trending_cache_time) < CACHE_5MIN:
        return _hf_trending_cache

    try:
        response = requests.get(HF_TRENDING_URL, timeout=15)
        response.raise_for_status()
        models = response.json()
        _hf_trending_cache = models
        _hf_trending_cache_time = now
        return models

    except Exception as e:
        if _hf_trending_cache is not None:
            print(f"Warning: Failed to refresh HF trending ({e}). Using stale cache.")
            return _hf_trending_cache
        raise RuntimeError(
            f"Could not fetch HF trending data and no cache available: {e}"
        )


# Pipeline tags for each AI modality and their human-readable names
HF_MODALITY_TAGS = [
    ("text-to-image", "Images"),
    ("text-to-speech", "Voice"),
    ("text-to-video", "Video"),
    ("text-to-audio", "Music / Audio"),
    ("image-to-video", "Animation"),
    ("image-to-3d", "3D"),
    ("automatic-speech-recognition", "Transcription"),
]


def get_hf_modality_champions():
    """
    Fetches the #1 model by downloads for each AI modality from HuggingFace.

    Makes 7 API calls (one per modality). Cached for 5 minutes.
    Returns a list of dicts with: tag, label, model_id, org, downloads, likes.
    Failed modalities are skipped (not included in the result).
    """
    global _hf_modality_cache, _hf_modality_cache_time

    now = time.time()
    if _hf_modality_cache is not None and (now - _hf_modality_cache_time) < CACHE_5MIN:
        return _hf_modality_cache

    results = []
    for tag, label in HF_MODALITY_TAGS:
        try:
            url = (
                f"https://huggingface.co/api/models?"
                f"sort=downloads&direction=-1&limit=1&pipeline_tag={tag}"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            models = resp.json()
            if models:
                m = models[0]
                model_id = m.get("id", "")
                org = model_id.split("/")[0] if "/" in model_id else "—"
                results.append({
                    "tag": tag,
                    "label": label,
                    "model_id": model_id,
                    "model_name": model_id.split("/", 1)[1] if "/" in model_id else model_id,
                    "org": org,
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                })
        except Exception as e:
            print(f"Warning: Failed to fetch HF modality '{tag}': {e}")

    _hf_modality_cache = results
    _hf_modality_cache_time = now
    return results


def get_swebench_leaderboard():
    """
    Fetches SWE-bench leaderboard from swebench.com.

    Parses the embedded JSON from the HTML page. Uses the "bash-only" category
    (most submissions, most comparable). Cached for 6 hours.

    Returns:
        List of dicts with: name, resolved (%), date, cost, os_model, os_system
    """
    global _swebench_cache, _swebench_cache_time
    import re
    import json

    now = time.time()
    if _swebench_cache is not None and (now - _swebench_cache_time) < CACHE_6H:
        return _swebench_cache

    try:
        response = requests.get(SWEBENCH_URL, timeout=15)
        response.raise_for_status()

        match = re.search(
            r'id="leaderboard-data"[^>]*>\s*(.*?)\s*</script>',
            response.text, re.DOTALL
        )
        if not match:
            raise ValueError("Could not find leaderboard-data in SWE-bench page")

        data = json.loads(match.group(1))
        # Use "bash-only" category — it has the most submissions
        bash_cat = next((d for d in data if d["name"] == "bash-only"), None)
        if not bash_cat:
            bash_cat = data[0]

        results = bash_cat["results"]
        _swebench_cache = results
        _swebench_cache_time = now
        return results

    except Exception as e:
        if _swebench_cache is not None:
            print(f"Warning: Failed to refresh SWE-bench ({e}). Using stale cache.")
            return _swebench_cache
        raise RuntimeError(
            f"Could not fetch SWE-bench data and no cache available: {e}"
        )


def get_pypi_downloads():
    """
    Fetches recent download counts for major AI SDK packages from PyPI Stats.

    Makes one request per package with 1.5s delay between to avoid rate limiting.
    Cached for 24 hours (download patterns don't change faster than that).

    Returns:
        List of dicts with: package, last_day, last_week, last_month
    """
    global _pypi_cache, _pypi_cache_time

    now = time.time()
    if _pypi_cache is not None and (now - _pypi_cache_time) < CACHE_24H:
        return _pypi_cache

    results = []
    for pkg in PYPI_PACKAGES:
        try:
            url = f"https://pypistats.org/api/packages/{pkg}/recent"
            resp = requests.get(url, timeout=10, headers={"User-Agent": "ai-race-dashboard/1.0"})
            if resp.status_code == 429:
                print(f"Warning: PyPI rate limited at '{pkg}', stopping batch.")
                break
            resp.raise_for_status()
            data = resp.json()["data"]
            results.append({
                "package": pkg,
                "last_day": data.get("last_day", 0),
                "last_week": data.get("last_week", 0),
                "last_month": data.get("last_month", 0),
            })
        except Exception as e:
            print(f"Warning: Failed to fetch PyPI stats for '{pkg}': {e}")
        time.sleep(3)

    if results:
        _pypi_cache = results
        _pypi_cache_time = now
        return results

    if _pypi_cache is not None:
        print("Warning: All PyPI requests failed. Using stale cache.")
        return _pypi_cache
    raise RuntimeError("Could not fetch PyPI data and no cache available")


def get_lmarena_leaderboard():
    """
    Fetches the latest LMArena (Chatbot Arena) leaderboard CSV from HuggingFace.

    Discovers the most recent leaderboard_table_YYYYMMDD.csv file from the
    space repository, then downloads and parses it. Cached for 6 hours.

    Returns:
        pd.DataFrame with columns from the CSV (Model, Arena Score, Organization, etc.)
    """
    global _lmarena_cache, _lmarena_cache_time

    now = time.time()
    if _lmarena_cache is not None and (now - _lmarena_cache_time) < CACHE_6H:
        return _lmarena_cache

    try:
        # List files to find the latest leaderboard CSV
        resp = requests.get(LMARENA_LEADERBOARD_URL, timeout=15)
        resp.raise_for_status()
        files = resp.json()

        # Find the most recent leaderboard_table file
        import re
        csv_files = [
            f["path"] for f in files
            if re.match(r'leaderboard_table_\d{8}\.csv', f["path"])
        ]
        if not csv_files:
            raise ValueError("No leaderboard CSV files found in space")

        latest_csv = sorted(csv_files)[-1]

        # Download the CSV
        csv_url = f"https://huggingface.co/spaces/lmarena-ai/arena-leaderboard/resolve/main/{latest_csv}"
        csv_resp = requests.get(csv_url, timeout=15)
        csv_resp.raise_for_status()

        import io
        df = pd.read_csv(io.StringIO(csv_resp.text))
        _lmarena_cache = df
        _lmarena_cache_time = now
        return df

    except Exception as e:
        if _lmarena_cache is not None:
            print(f"Warning: Failed to refresh LMArena ({e}). Using stale cache.")
            return _lmarena_cache
        raise RuntimeError(
            f"Could not fetch LMArena data and no cache available: {e}"
        )
