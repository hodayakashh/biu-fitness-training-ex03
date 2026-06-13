"""Unit tests for data_clusters — per-action muscle profiles and data-driven labels."""

import numpy as np
import pandas as pd
import pytest

from fitness_rl.services.data_clusters import (
    compute_action_profiles,
    describe_clusters,
    muscle_columns,
)


@pytest.fixture()
def daily_df() -> pd.DataFrame:
    """Three clusters with one-hot muscle profiles and increasing volume."""
    return pd.DataFrame(
        {
            "action_label": [0, 0, 1, 1, 2],
            "total_volume": [10.0, 10.0, 100.0, 100.0, 1000.0],
            "mg_chest": [1.0, 1.0, 0.0, 0.0, 0.0],
            "mg_legs": [0.0, 0.0, 1.0, 1.0, 0.0],
            "mg_cardio": [0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )


class TestMuscleColumns:
    def test_sorted_mg_columns(self, daily_df):
        assert muscle_columns(daily_df) == ["mg_cardio", "mg_chest", "mg_legs"]

    def test_no_mg_columns(self):
        assert muscle_columns(pd.DataFrame({"a": [1]})) == []


class TestComputeActionProfiles:
    def test_shape_and_normalisation(self, daily_df):
        profiles, mg_cols = compute_action_profiles(daily_df, n_actions=3)
        assert profiles.shape == (3, 3)
        np.testing.assert_allclose(profiles.sum(axis=1), 1.0, atol=1e-6)
        assert mg_cols == ["mg_cardio", "mg_chest", "mg_legs"]

    def test_dominant_muscle_per_cluster(self, daily_df):
        profiles, mg_cols = compute_action_profiles(daily_df, n_actions=3)
        # cluster 0 → chest, 1 → legs, 2 → cardio (indices into sorted mg_cols)
        assert mg_cols[int(profiles[0].argmax())] == "mg_chest"
        assert mg_cols[int(profiles[1].argmax())] == "mg_legs"
        assert mg_cols[int(profiles[2].argmax())] == "mg_cardio"

    def test_empty_cluster_falls_back_to_uniform(self, daily_df):
        """A cluster with no assigned days gets a uniform (max-entropy) profile."""
        profiles, _ = compute_action_profiles(daily_df, n_actions=4)
        np.testing.assert_allclose(profiles[3], 1.0 / 3.0, atol=1e-6)

    def test_no_muscle_columns_uniform(self):
        df = pd.DataFrame({"action_label": [0, 1], "total_volume": [1.0, 2.0]})
        profiles, mg_cols = compute_action_profiles(df, n_actions=2)
        assert mg_cols == []
        assert profiles.shape == (2, 1)
        np.testing.assert_allclose(profiles, 1.0)

    def test_missing_action_label_uniform(self):
        df = pd.DataFrame({"mg_chest": [1.0], "mg_legs": [0.0]})
        profiles, _ = compute_action_profiles(df, n_actions=2)
        np.testing.assert_allclose(profiles, 0.5)


class TestDescribeClusters:
    def test_label_format_and_tiers(self, daily_df):
        profiles, mg_cols = compute_action_profiles(daily_df, n_actions=3)
        labels = describe_clusters(daily_df, profiles, mg_cols, n_actions=3)
        assert labels[0] == "[0] Chest (low)"
        assert labels[1] == "[1] Legs (moderate)"
        assert labels[2] == "[2] Cardio (high)"

    def test_labels_unique_and_complete(self, daily_df):
        profiles, mg_cols = compute_action_profiles(daily_df, n_actions=3)
        labels = describe_clusters(daily_df, profiles, mg_cols, n_actions=3)
        assert set(labels.keys()) == {0, 1, 2}
        assert len(set(labels.values())) == 3

    def test_mixed_when_no_muscle_columns(self):
        df = pd.DataFrame({"action_label": [0, 1], "total_volume": [1.0, 2.0]})
        profiles, mg_cols = compute_action_profiles(df, n_actions=2)
        labels = describe_clusters(df, profiles, mg_cols, n_actions=2)
        assert "Mixed" in labels[0]

    def test_missing_total_volume_defaults_low(self):
        df = pd.DataFrame({"action_label": [0], "mg_chest": [1.0], "mg_legs": [0.0]})
        profiles, mg_cols = compute_action_profiles(df, n_actions=1)
        labels = describe_clusters(df, profiles, mg_cols, n_actions=1)
        assert labels[0] == "[0] Chest (low)"
