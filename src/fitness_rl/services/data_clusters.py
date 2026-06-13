"""
Cluster post-processing helpers — per-action muscle profiles and data-driven labels.

Separated from DataPreprocessor so each file stays under 150 lines (BIU §3.1).

K-Means cluster IDs are arbitrary, so workout names are NOT hardcoded against fixed
IDs (review finding #3). Instead each cluster is described from its OWN data: the
dominant muscle group (argmax of its mean ``mg_*`` profile) plus a load tier derived
from the cluster's mean ``total_volume``. The same per-cluster muscle profiles feed
the action-conditioned muscle-balance dynamics of the RL environment (PLAN ADR-001).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TIERS = ("low", "moderate", "high")


def muscle_columns(daily_df: pd.DataFrame) -> list[str]:
    """Return the sorted ``mg_*`` muscle-group column names present in the summaries."""
    return sorted(c for c in daily_df.columns if c.startswith("mg_"))


def compute_action_profiles(daily_df: pd.DataFrame, n_actions: int) -> tuple[np.ndarray, list[str]]:
    """
    Mean muscle-group distribution per action cluster.

    Each cluster's profile is the average of the per-day normalised muscle
    distribution over the days assigned to that cluster. Rows are renormalised to
    sum to 1; empty clusters (or data with no muscle columns) fall back to a uniform
    distribution so downstream entropy is always well-defined.

    Args:
        daily_df:  Daily summaries with ``action_label`` and ``mg_*`` columns.
        n_actions: Number of discrete action clusters.

    Returns:
        Tuple ``(profiles, mg_cols)`` where ``profiles`` is a float32 array of shape
        ``(n_actions, n_mg)`` whose rows sum to 1, and ``mg_cols`` are the column names.
    """
    mg_cols = muscle_columns(daily_df)
    n_mg = max(len(mg_cols), 1)
    profiles = np.full((n_actions, n_mg), 1.0 / n_mg, dtype=np.float64)
    if not mg_cols or "action_label" not in daily_df.columns:
        return profiles.astype(np.float32), mg_cols

    grouped = daily_df.groupby("action_label")[mg_cols].mean()
    for action in range(n_actions):
        if action not in grouped.index:
            continue
        row = grouped.loc[action].to_numpy(dtype=np.float64).clip(min=0.0)
        total = row.sum()
        if total > 0:
            profiles[action] = row / total
    return profiles.astype(np.float32), mg_cols


def describe_clusters(
    daily_df: pd.DataFrame,
    profiles: np.ndarray,
    mg_cols: list[str],
    n_actions: int,
) -> dict[int, str]:
    """
    Build a grounded, human-readable label per cluster.

    Label format: ``"<Dominant muscle> (<load tier>)"`` — e.g. ``"Quadriceps (high)"``.
    The dominant muscle is the argmax of the cluster's mean profile; the load tier is
    the cluster's mean ``total_volume`` placed into low/moderate/high terciles across
    all clusters. A leading ``[id]`` keeps labels unique when two clusters share a
    description.

    Args:
        daily_df:  Daily summaries with ``action_label`` and ``total_volume``.
        profiles:  Output of :func:`compute_action_profiles`.
        mg_cols:   Muscle-group column names aligned to ``profiles`` columns.
        n_actions: Number of discrete action clusters.

    Returns:
        Mapping ``{cluster_id: label}`` for every cluster in ``range(n_actions)``.
    """
    volumes = _mean_volumes(daily_df, n_actions)
    tiers = _load_tiers(volumes)
    labels: dict[int, str] = {}
    for action in range(n_actions):
        if mg_cols and float(profiles[action].max()) > 0.0:
            dominant = mg_cols[int(np.argmax(profiles[action]))].removeprefix("mg_")
        else:
            dominant = "mixed"
        labels[action] = f"[{action}] {dominant.title()} ({tiers[action]})"
    return labels


def _mean_volumes(daily_df: pd.DataFrame, n_actions: int) -> np.ndarray:
    """Mean ``total_volume`` per cluster (0 for clusters with no assigned days)."""
    if "total_volume" not in daily_df.columns or "action_label" not in daily_df.columns:
        return np.zeros(n_actions, dtype=np.float64)
    means = daily_df.groupby("action_label")["total_volume"].mean()
    return np.array([float(means.get(action, 0.0)) for action in range(n_actions)])


def _load_tiers(volumes: np.ndarray) -> list[str]:
    """Map each cluster's mean volume to a low/moderate/high tercile label."""
    q1, q2 = np.quantile(volumes, [1 / 3, 2 / 3])
    tiers: list[str] = []
    for value in volumes:
        if value <= q1:
            tiers.append(_TIERS[0])
        elif value <= q2:
            tiers.append(_TIERS[1])
        else:
            tiers.append(_TIERS[2])
    return tiers
