"""Alignment measures between two feature sets on the same N items.

mutual_knn: the PRH paper's measure. For each item, the overlap between its k
nearest neighbours in space A and in space B, averaged. 1 means identical
neighbourhoods; about k/(N-1) means unrelated.
linear_cka: centred kernel alignment with a linear kernel (Kornblith et al. 2019).
"""
import numpy as np


def _normalise(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def knn_indices(x, k):
    x = _normalise(x)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)[:, :k]


def mutual_knn(a, b, k=10):
    if len(a) != len(b):
        raise ValueError("a and b must describe the same items")
    if not 0 < k < len(a):
        raise ValueError("k must be between 1 and N-1")
    ia, ib = knn_indices(a, k), knn_indices(b, k)
    return float(np.mean([len(set(ra) & set(rb)) / k for ra, rb in zip(ia, ib)]))


def linear_cka(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(0); b = b - b.mean(0)
    num = np.linalg.norm(b.T @ a, "fro") ** 2
    den = np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro")
    return float(num / den)


def shuffled_baseline(a, b, k=10, seed=0):
    perm = np.random.default_rng(seed).permutation(len(b))
    return mutual_knn(a, np.asarray(b)[perm], k)


def bootstrap(a, b, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    vals = []
    for _ in range(n):
        idx = np.unique(rng.integers(0, len(a), len(a)))
        vals.append(fn(a[idx], b[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
