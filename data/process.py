"""
process.py — Transforms raw benchmark data into the heatmap structure.

Why this module exists:
This is where the "ranking intelligence" lives. It takes 1,462 raw benchmark
scores and distills them into a clean top-8 ranking across 5 human-readable
categories. The logic is entirely dynamic — no model names are hardcoded.

Key concept — Benchmark-to-Category mapping:
The raw CSV has benchmark names like "MATH level 5" or "GPQA diamond".
Humans think in categories like "Math" or "Science". This module bridges
that gap with a simple dictionary mapping.
"""

import pandas as pd

# ── Benchmark → Category mapping ────────────────────────────────────
# Why these specific benchmarks?
# These were selected because they're widely reported, cover distinct
# capabilities, and are present for most top models in the Epoch dataset.
# If Epoch adds new benchmarks, you can add them to these lists.
CATEGORY_MAP = {
    "Math": ["MATH level 5", "OTIS Mock AIME 2024-2025"],
    "Code": ["WeirdML"],
    "Reasoning": ["ARC-AGI-2"],
    "Science": ["GPQA diamond"],
    "Common Sense": ["SimpleBench"],
}


def build_heatmap_data(df):
    """
    Takes raw benchmark DataFrame and returns a clean matrix for the heatmap.

    Step-by-step logic:
    1. Filter the DataFrame to only include benchmarks we care about.
    2. Map each benchmark to its human-readable category.
    3. For each model_group + category, take the BEST score.
       (Why best? A model_group may have multiple entries per benchmark.
        We want its peak performance.)
    4. Filter to models that have scores in at least 4 of 5 categories.
       (Why 4? Requiring all 5 would exclude too many models. Requiring
        fewer would let incomplete models sneak into the ranking.)
    5. Rank by average score across categories, take top 8.

    Args:
        df: Raw DataFrame from fetch.get_benchmarks()

    Returns:
        heatmap_df: DataFrame where rows=models, columns=categories, values=scores
        last_date: The most recent date in the data (for the "last updated" label)
    """
    # Step 1: Create a reverse mapping — benchmark name → category name
    benchmark_to_category = {}
    for category, benchmarks in CATEGORY_MAP.items():
        for bench in benchmarks:
            benchmark_to_category[bench] = category

    # Filter to only the benchmarks we mapped
    known_benchmarks = list(benchmark_to_category.keys())
    filtered = df[df["benchmark"].isin(known_benchmarks)].copy()

    if filtered.empty:
        raise ValueError(
            "No matching benchmarks found in data. "
            f"Expected benchmarks: {known_benchmarks}. "
            f"Available benchmarks: {df['benchmark'].unique().tolist()[:20]}"
        )

    # Step 2: Add category column
    filtered["category"] = filtered["benchmark"].map(benchmark_to_category)

    # Step 3: Best score per model_group per category
    # Why model_group instead of model? Because "GPT-5.4 Pro" might appear
    # as multiple rows (different dates/runs). model_group clusters them.
    best_scores = (
        filtered.groupby(["model_group", "category"])["performance"]
        .max()
        .reset_index()
    )

    # Step 4: Count how many categories each model has scores for
    category_count = (
        best_scores.groupby("model_group")["category"]
        .nunique()
        .reset_index(name="num_categories")
    )
    total_categories = len(CATEGORY_MAP)
    # Keep models with scores in at least 4 of 5 categories
    min_categories = total_categories - 1
    qualified = category_count[category_count["num_categories"] >= min_categories]
    qualified_models = set(qualified["model_group"])

    best_scores = best_scores[best_scores["model_group"].isin(qualified_models)]

    # Step 5: Rank by average score, take top 8
    avg_score = (
        best_scores.groupby("model_group")["performance"]
        .mean()
        .reset_index(name="avg")
        .sort_values("avg", ascending=False)
    )
    top_models = avg_score.head(8)["model_group"].tolist()

    # Build the final matrix: rows=models (ranked), columns=categories
    top_scores = best_scores[best_scores["model_group"].isin(top_models)]
    heatmap_df = top_scores.pivot(
        index="model_group", columns="category", values="performance"
    )

    # Reorder rows by ranking (best average first)
    heatmap_df = heatmap_df.loc[top_models]

    # Reorder columns in our preferred display order
    column_order = [c for c in CATEGORY_MAP.keys() if c in heatmap_df.columns]
    heatmap_df = heatmap_df[column_order]

    # Convert from 0-1 scale to percentage (0-100) for display
    heatmap_df = heatmap_df * 100

    # Get the most recent date in the data for the "last updated" label
    last_date = filtered["date"].max() if "date" in filtered.columns else "Unknown"

    return heatmap_df, last_date


# ── Graph 2: Pricing scatter ──────────────────────────────────────

# The top providers we want to highlight with distinct colors.
# All others get a muted "Other" color so the chart isn't overwhelming.
TOP_ORGS = ["openai", "anthropic", "google", "meta-llama", "mistralai",
            "deepseek", "x-ai", "qwen"]


def _extract_family(model_id):
    """
    Extracts the model family from an OpenRouter model ID by stripping
    date suffixes and version tags.

    Examples:
        "openai/gpt-4o-2024-11-20"  → "openai/gpt-4o"
        "openai/gpt-4o-mini"        → "openai/gpt-4o-mini"
        "anthropic/claude-3.5-sonnet" → "anthropic/claude-3.5-sonnet"
        "google/gemini-2.5-pro-preview-05-06" → "google/gemini-2.5-pro-preview"
        "openai/gpt-4o:extended"    → "openai/gpt-4o:extended"

    The pattern: strip trailing segments that look like dates (YYYY-MM-DD,
    YYYY-MM, MM-DD) from the end of the ID.
    """
    import re
    # Strip date-like suffixes at the end: -2024-11-20, -05-06, -2024-08
    cleaned = re.sub(r'(-\d{4})?-\d{2}-\d{2}$', '', model_id)
    # Also catch standalone year-month: -2024-11
    cleaned = re.sub(r'-\d{4}-\d{2}$', '', cleaned)
    return cleaned


def build_pricing_data(raw_models):
    """
    Transforms raw OpenRouter JSON into a DataFrame for the scatter plot.

    Steps:
    1. Extract org from model id — keep only TOP_ORGS.
    2. Convert per-token prices to per-1M-token dollars.
    3. Filter out free models and extreme outliers (>$100/1M).
    4. Deduplicate model families — keep the most recently created
       version per family (e.g., gpt-4o-2024-11-20 beats gpt-4o-2024-08-06).

    Returns:
        pd.DataFrame with columns: name, org, org_display, prompt_1m,
        completion_1m, context_length, family
    """
    rows = []
    for m in raw_models:
        org = m["id"].split("/")[0]

        # Filter 1: Top orgs only
        if org not in TOP_ORGS:
            continue

        pricing = m.get("pricing", {})
        prompt_raw = float(pricing.get("prompt", "0") or "0")
        completion_raw = float(pricing.get("completion", "0") or "0")

        if prompt_raw <= 0 or completion_raw <= 0:
            continue

        prompt_1m = prompt_raw * 1_000_000
        completion_1m = completion_raw * 1_000_000

        if prompt_1m > 100 or completion_1m > 100:
            continue

        rows.append({
            "id": m["id"],
            "name": m.get("name", m["id"]),
            "org": org,
            "org_display": org,
            "family": _extract_family(m["id"]),
            "prompt_1m": round(prompt_1m, 3),
            "completion_1m": round(completion_1m, 3),
            "context_length": m.get("context_length", 0),
            "created": m.get("created", 0),
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Filter 2: Drop models older than 12 months — removes legacy
    # models (GPT-3.5, GPT-4 v0314, etc.) that clutter the chart
    import time
    twelve_months_ago = time.time() - (365 * 86400)
    df = df[df["created"] >= twelve_months_ago]

    # Filter 3: Deduplicate — keep most recent per family
    df = df.sort_values("created", ascending=False)
    df = df.drop_duplicates(subset="family", keep="first")
    df = df.drop(columns=["id", "created", "family"])
    df = df.reset_index(drop=True)

    return df


# ── Graph 3: Downloads by org ─────────────────────────────────────

# Friendly display names for orgs on HuggingFace
ORG_DISPLAY_NAMES = {
    "Qwen": "Qwen (Alibaba)",
    "meta-llama": "Meta",
    "openai-community": "OpenAI (community)",
    "openai": "OpenAI",
    "deepseek-ai": "DeepSeek",
    "nvidia": "NVIDIA",
    "microsoft": "Microsoft",
    "mistralai": "Mistral",
    "google": "Google",
    "facebook": "Meta (legacy)",
    "EleutherAI": "EleutherAI",
    "HuggingFaceTB": "HuggingFace",
}


def build_downloads_data(raw_models, top_n=12):
    """
    Aggregates HuggingFace model downloads by organization.

    Steps:
    1. Extract org from model id (e.g., "Qwen/Qwen3-8B" → "Qwen").
    2. Sum downloads per org.
    3. Take top N orgs by total downloads.
    4. Apply friendly display names.

    Returns:
        pd.DataFrame with columns: org, display_name, downloads, model_count
        Sorted descending by downloads.
    """
    from collections import defaultdict

    org_downloads = defaultdict(int)
    org_count = defaultdict(int)

    for m in raw_models:
        model_id = m.get("id", "")
        org = model_id.split("/")[0] if "/" in model_id else "community"

        # Skip test/internal repos
        if org in ("trl-internal-testing", "dphn", "hmellor", "community"):
            continue

        org_downloads[org] += m.get("downloads", 0)
        org_count[org] += 1

    # Build DataFrame, sort, take top N
    rows = [
        {
            "org": org,
            "display_name": ORG_DISPLAY_NAMES.get(org, org),
            "downloads": downloads,
            "model_count": org_count[org],
        }
        for org, downloads in org_downloads.items()
    ]

    df = pd.DataFrame(rows)
    df = df.sort_values("downloads", ascending=False).head(top_n)
    df = df.reset_index(drop=True)

    return df


# ── Graph 4: Trending models ─────────────────────────────────────

def build_trending_data(raw_models):
    """
    Transforms raw HuggingFace trending JSON into a clean DataFrame.

    Steps:
    1. Extract org, model short name, likes, downloads, creation date.
    2. Compute "days ago" from createdAt for freshness display.

    Returns:
        pd.DataFrame with columns: rank, model_name, org, likes, downloads,
        created_date, days_ago
    """
    from datetime import datetime

    rows = []
    now = datetime.utcnow()

    for i, m in enumerate(raw_models, 1):
        model_id = m.get("id", "")
        org = model_id.split("/")[0] if "/" in model_id else "—"
        # Short name: drop the org prefix
        short_name = model_id.split("/", 1)[1] if "/" in model_id else model_id

        created_str = m.get("createdAt", "")[:10]  # "2026-03-18"
        try:
            created_dt = datetime.strptime(created_str, "%Y-%m-%d")
            days_ago = (now - created_dt).days
        except ValueError:
            days_ago = None

        rows.append({
            "rank": i,
            "model_name": short_name,
            "org": org,
            "likes": m.get("likes", 0),
            "downloads": m.get("downloads", 0),
            "created_date": created_str,
            "days_ago": days_ago,
        })

    return pd.DataFrame(rows)


# ── Graph 5: Timeline of models by company ────────────────────────

# Consolidate org variants into clean names
TIMELINE_ORG_MAP = {
    "Google": "Google",
    "Google DeepMind": "Google",
    "Google Research": "Google",
    "Google Brain": "Google",
    "DeepMind": "Google",
    "Meta AI": "Meta",
    "Facebook AI Research": "Meta",
    "Facebook": "Meta",
    "OpenAI": "OpenAI",
    "Anthropic": "Anthropic",
    "Alibaba": "Alibaba",
    "Microsoft": "Microsoft",
    "DeepSeek": "DeepSeek",
    "xAI": "xAI",
    "Baidu": "Baidu",
    "ByteDance": "ByteDance",
    "NVIDIA": "NVIDIA",
    "Mistral AI": "Mistral",
}

TIMELINE_ORGS = list(dict.fromkeys(TIMELINE_ORG_MAP.values()))


def build_timeline_data(df):
    """
    Transforms Epoch notable models into a timeline dataset.

    Steps:
    1. Parse publication dates, filter to 2020+.
    2. Normalize org names (merge Google/DeepMind variants, etc.).
    3. Keep only models from top orgs.
    4. Include training compute (FLOP) and parameters where available.

    Returns:
        pd.DataFrame with columns: model, org, date, compute_flop,
        parameters, has_compute
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["Publication date"], errors="coerce")
    df = df[df["date"] >= "2020-01-01"]

    # Normalize orgs
    df["org"] = df["Organization"].map(TIMELINE_ORG_MAP)
    df = df[df["org"].notna()]

    # Build clean output
    result = pd.DataFrame({
        "model": df["Model"],
        "org": df["org"],
        "date": df["date"],
        "compute_flop": pd.to_numeric(df["Training compute (FLOP)"], errors="coerce"),
        "parameters": pd.to_numeric(df["Parameters"], errors="coerce"),
    })

    result["has_compute"] = result["compute_flop"].notna()

    return result.reset_index(drop=True)


# ── Graph 7: Capabilities heatmap ─────────────────────────────────

# Friendly display names for OpenRouter orgs
CAPABILITY_ORG_NAMES = {
    "openai": "OpenAI",
    "google": "Google",
    "qwen": "Qwen",
    "anthropic": "Anthropic",
    "mistralai": "Mistral",
    "meta-llama": "Meta",
    "z-ai": "Zhipu AI",
    "nvidia": "NVIDIA",
    "deepseek": "DeepSeek",
    "x-ai": "xAI",
}

CAPABILITY_COLUMNS = ["Vision", "Audio in", "Video in", "Files", "Image gen", "Audio gen"]


def build_capabilities_data(raw_models, top_n=10):
    """
    Parses OpenRouter model architectures to build a capabilities matrix.

    How architecture parsing works:
    - Each model has architecture.input_modalities (array of strings like
      ["text", "image", "audio", "video", "file"])
    - And architecture.output_modalities (array like ["text", "image", "audio"])
    - We check membership in these arrays to determine capabilities.

    Steps:
    1. For each model, extract org and check input/output modalities.
    2. Aggregate: count how many models per org have each capability.
    3. Take top N orgs by total model count.
    4. Return as a DataFrame (rows=orgs, columns=capabilities).
    """
    from collections import defaultdict

    org_caps = defaultdict(lambda: defaultdict(int))
    org_total = defaultdict(int)

    for m in raw_models:
        org = m["id"].split("/")[0]
        arch = m.get("architecture", {}) or {}
        inputs = arch.get("input_modalities", []) or []
        outputs = arch.get("output_modalities", []) or []

        org_total[org] += 1

        if "image" in inputs:
            org_caps[org]["Vision"] += 1
        if "audio" in inputs:
            org_caps[org]["Audio in"] += 1
        if "video" in inputs:
            org_caps[org]["Video in"] += 1
        if "file" in inputs:
            org_caps[org]["Files"] += 1
        if "image" in outputs:
            org_caps[org]["Image gen"] += 1
        if "audio" in outputs:
            org_caps[org]["Audio gen"] += 1

    # Top N orgs by total model count
    top_orgs = sorted(org_total.items(), key=lambda x: -x[1])[:top_n]

    rows = []
    for org, total in top_orgs:
        display = CAPABILITY_ORG_NAMES.get(org, org)
        row = {"org": org, "display_name": display, "total_models": total}
        for cap in CAPABILITY_COLUMNS:
            row[cap] = org_caps[org].get(cap, 0)
        rows.append(row)

    return pd.DataFrame(rows)


# ── Graph 8: Modality champions ───────────────────────────────────

# Friendly org names for HuggingFace modality champions
MODALITY_ORG_NAMES = {
    "stabilityai": "Stability AI",
    "hexgrad": "Hexgrad",
    "Wan-AI": "Wan-AI",
    "facebook": "Meta",
    "lightx2v": "LightX2V",
    "microsoft": "Microsoft",
    "pyannote": "Pyannote",
}


def build_modality_champions(raw_results):
    """
    Transforms raw HuggingFace modality champion data into a DataFrame.

    The data comes pre-processed from fetch — one champion per modality.
    We just clean up org names and format downloads.

    Returns:
        pd.DataFrame with columns: label, model_name, org, display_org,
        downloads, likes
    """
    rows = []
    for r in raw_results:
        display_org = MODALITY_ORG_NAMES.get(r["org"], r["org"])
        rows.append({
            "label": r["label"],
            "model_name": r["model_name"],
            "org": r["org"],
            "display_org": display_org,
            "downloads": r["downloads"],
            "likes": r["likes"],
        })

    return pd.DataFrame(rows)
