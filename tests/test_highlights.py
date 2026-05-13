"""Tests for sentrysearch.highlights."""

import math

import numpy as np
import pytest

from sentrysearch.highlights import (
    _dedupe_indices,
    _exclude_baseline_mask,
    _normalize,
    _score_centroid,
    _score_knn,
    _score_lof,
    rank_highlights,
)


def _cluster_with_outliers(rng: np.random.Generator, dim: int = 16) -> np.ndarray:
    """20 points clustered near +x axis, plus 3 outliers along -x, +y, +z."""
    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0
    cluster = base + 0.02 * rng.standard_normal((20, dim)).astype(np.float32)
    outliers = np.zeros((3, dim), dtype=np.float32)
    outliers[0, 0] = -1.0
    outliers[1, 1] = 1.0
    outliers[2, 2] = 1.0
    return np.vstack([cluster, outliers])


def _add_chunk(store, idx: int, embedding: np.ndarray, source: str = "v.mp4",
               start: float = 0.0) -> None:
    store.add_chunk(
        chunk_id=f"c{idx:03d}",
        embedding=embedding.tolist(),
        metadata={
            "source_file": source,
            "start_time": float(start),
            "end_time": float(start + 30.0),
        },
    )


class TestScoring:
    def test_centroid_ranks_outliers_high(self):
        rng = np.random.default_rng(0)
        X = _cluster_with_outliers(rng)
        Xn = _normalize(X)
        scores = _score_centroid(Xn)
        # The 3 known outliers are the last 3 rows
        top3 = set(np.argsort(-scores)[:3].tolist())
        assert top3 == {20, 21, 22}

    def test_knn_ranks_outliers_high(self):
        rng = np.random.default_rng(0)
        X = _cluster_with_outliers(rng)
        Xn = _normalize(X)
        scores = _score_knn(Xn, k=5)
        top3 = set(np.argsort(-scores)[:3].tolist())
        assert top3 == {20, 21, 22}

    def test_lof_ranks_outliers_high(self):
        rng = np.random.default_rng(0)
        X = _cluster_with_outliers(rng)
        Xn = _normalize(X)
        scores = _score_lof(Xn, k=5)
        top3 = set(np.argsort(-scores)[:3].tolist())
        assert top3 == {20, 21, 22}

    def test_knn_clamps_k_above_n(self):
        Xn = _normalize(np.random.default_rng(1).standard_normal((4, 8)).astype(np.float32))
        # k larger than n-1 should not raise
        scores = _score_knn(Xn, k=100)
        assert scores.shape == (4,)


class TestDedupe:
    def test_drops_near_duplicates(self):
        # Two near-identical points + one distinct point
        Xn = _normalize(np.array([
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],   # near-dup of index 0
            [0.0, 1.0, 0.0],     # distinct
        ], dtype=np.float32))
        ranked = np.array([0, 1, 2])
        kept = _dedupe_indices(ranked, Xn, threshold=0.9, limit=10)
        assert kept == [0, 2]

    def test_respects_limit(self):
        Xn = _normalize(np.eye(5, dtype=np.float32))
        ranked = np.arange(5)
        kept = _dedupe_indices(ranked, Xn, threshold=0.5, limit=2)
        assert len(kept) == 2


class TestBaselineMask:
    def test_keeps_far_points(self):
        rng = np.random.default_rng(2)
        X = _cluster_with_outliers(rng)
        Xn = _normalize(X)
        mask = _exclude_baseline_mask(Xn)
        # All three explicit outliers should survive
        assert mask[20] and mask[21] and mask[22]
        # Roughly half are kept
        assert 0.3 * len(mask) <= mask.sum() <= 0.7 * len(mask)

    def test_tiny_index_passthrough(self):
        Xn = _normalize(np.eye(3, dtype=np.float32))
        mask = _exclude_baseline_mask(Xn)
        assert mask.all()


class TestRankHighlights:
    def _populate(self, store, rng=None):
        rng = rng or np.random.default_rng(0)
        X = _cluster_with_outliers(rng)
        # Pad to 768 dims to match the default Chroma collection dimensionality
        # (the store doesn't actually enforce a dim, but be consistent with prod).
        pad = np.zeros((X.shape[0], 768 - X.shape[1]), dtype=np.float32)
        X_full = np.hstack([X, pad])
        for i, vec in enumerate(X_full):
            _add_chunk(store, i, vec, start=i * 30.0)
        return X_full

    def test_returns_outliers(self, tmp_store):
        self._populate(tmp_store)
        results = rank_highlights(tmp_store, count=3, method="knn", neighbors=5,
                                  dedupe_threshold=1.0)
        # Expect the three planted outliers — start_times 20*30, 21*30, 22*30
        start_times = {r["start_time"] for r in results}
        assert start_times == {600.0, 630.0, 660.0}

    def test_count_caps_results(self, tmp_store):
        self._populate(tmp_store)
        results = rank_highlights(tmp_store, count=2, method="knn", neighbors=5,
                                  dedupe_threshold=1.0)
        assert len(results) == 2

    def test_empty_index_returns_empty(self, tmp_store):
        results = rank_highlights(tmp_store, count=5)
        assert results == []

    def test_invalid_method_raises(self, tmp_store):
        with pytest.raises(ValueError):
            rank_highlights(tmp_store, count=1, method="bogus")

    def test_invalid_against_mode_raises(self, tmp_store):
        with pytest.raises(ValueError):
            rank_highlights(tmp_store, count=1, against_mode="elsewhere")

    def test_against_within_restricts_to_query_neighborhood(self, tmp_store):
        X_full = self._populate(tmp_store)
        # Query close to the cluster — "within" should rank anomalies *inside*
        # the matches of that query, i.e. the points furthest from the cluster
        # center while still in the top-matches pool.
        query = X_full[0].copy()  # near cluster center
        results = rank_highlights(
            tmp_store, count=3, method="knn", neighbors=3,
            dedupe_threshold=1.0,
            against_embedding=query,
            against_mode="within",
            against_pool=20,  # smaller than full index to actually restrict
        )
        # All returned chunks must be from the cluster (start_time < 600)
        # because the pool is the 20 nearest to the cluster centroid.
        assert all(r["start_time"] < 600.0 for r in results)

    def test_against_global_includes_outliers(self, tmp_store):
        X_full = self._populate(tmp_store)
        # Query aimed at one specific outlier (+y direction, row 21).
        query = X_full[21].copy()
        results = rank_highlights(
            tmp_store, count=1, method="knn", neighbors=5,
            dedupe_threshold=1.0,
            against_embedding=query,
            against_mode="global",
        )
        # The top hit should be that outlier (start_time = 21 * 30)
        assert results[0]["start_time"] == 630.0
