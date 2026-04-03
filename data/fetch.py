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

CACHE_24H = 86400
CACHE_5MIN = 300

BENCHMARKS_URL = "https://epoch.ai/data/eci_benchmarks.csv"
NOTABLE_URL = "https://epoch.ai/data/notable_ai_models.csv"
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
HF_DOWNLOADS_URL = "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=200&pipeline_tag=text-generation"
HF_TRENDING_URL = "https://huggingface.co/api/models?sort=trendingScore&direction=-1&limit=15&pipeline_tag=text-generation"


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
