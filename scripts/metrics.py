"""Alignment measures between two feature sets on the same N items.

mutual_knn: the PRH paper's measure. For each item, the overlap between its k
nearest neighbours in space A and in space B, averaged. 1 means identical
neighbourhoods; about k/(N-1) means unrelated.
linear_cka: centred kernel alignment with a linear kernel (Kornblith et al. 2019).
unbiased_cka: linear CKA with the unbiased HSIC estimator (Song et al. 2012),
as used by Huh et al. (2024) and Balestriero and LeCun (2026).
prh_prep / prh_score: the PRH paper's 0.95-quantile elementwise clipping, L2
normalisation and max over layer pairs.
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


def per_item_mutual_knn(a, b, k=10):
    if len(a) != len(b):
        raise ValueError("a and b must describe the same items")
    if not 0 < k < len(a):
        raise ValueError("k must be between 1 and N-1")
    ia, ib = knn_indices(a, k), knn_indices(b, k)
    return np.array([len(set(ra) & set(rb)) / k for ra, rb in zip(ia, ib)], dtype=np.float64)


def mutual_knn(a, b, k=10):
    return float(np.mean(per_item_mutual_knn(a, b, k)))


def linear_cka(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(0)
    b = b - b.mean(0)
    num = np.linalg.norm(b.T @ a, "fro") ** 2
    den = np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro")
    return float(num / den)


def unbiased_cka(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b) or len(a) < 4:
        raise ValueError("unbiased_cka needs at least 4 paired items")

    def hsic(k_mat, l_mat):
        m = len(k_mat)
        k_zero = k_mat.copy()
        l_zero = l_mat.copy()
        np.fill_diagonal(k_zero, 0.0)
        np.fill_diagonal(l_zero, 0.0)
        term1 = np.sum(k_zero * l_zero)
        term2 = np.sum(k_zero) * np.sum(l_zero) / ((m - 1) * (m - 2))
        term3 = 2.0 * float(np.sum(k_zero, axis=0) @ np.sum(l_zero, axis=1)) / (m - 2)
        return (term1 + term2 - term3) / (m * (m - 3))

    k_mat = a @ a.T
    l_mat = b @ b.T
    denom = np.sqrt(max(hsic(k_mat, k_mat) * hsic(l_mat, l_mat), 1e-30))
    return float(hsic(k_mat, l_mat) / denom)


def prh_prep(x, q=0.95):
    x = np.asarray(x, dtype=np.float32)
    flat = np.abs(x).ravel()
    stride = max(1, -(-flat.size // 10_000_000))
    thresh = float(np.quantile(flat[::stride], q))
    clipped = np.clip(x, -thresh, thresh)
    return _normalise(clipped)


def prh_score(layers_a, layers_b, k=10, metric="mknn"):
    if metric not in ("mknn", "cka"):
        raise ValueError(f"unknown metric {metric!r}")
    fn = (lambda a, b: mutual_knn(a, b, k)) if metric == "mknn" else unbiased_cka
    fa = [prh_prep(layer) for layer in layers_a]
    fb = [prh_prep(layer) for layer in layers_b]
    best = (-np.inf, -1, -1)
    for i, a in enumerate(fa):
        for j, b in enumerate(fb):
            s = fn(a, b)
            if s > best[0]:
                best = (float(s), i, j)
    return best


def shuffled_baseline(a, b, k=10, seed=0):
    perm = np.random.default_rng(seed).permutation(len(b))
    return mutual_knn(a, np.asarray(b)[perm], k)


def bootstrap(a, b, fn=None, n=1000, seed=0, k=10):
    """95% bootstrap interval over the N evaluation items.

    When fn is None, computes each item's k-NN overlap on the full N-item graph
    once, then resamples those N item overlaps with replacement. This keeps the
    neighbourhood pool size fixed at N and avoids duplicate-point neighbours.
    When a custom fn(a, b) is given, draws subsets of size round(0.8 * N)
    without replacement and rescales their spread by sqrt(size / (N - size)).
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    if fn is None:
        overlaps = per_item_mutual_knn(a, b, k)
        draws = rng.integers(0, len(overlaps), size=(n, len(overlaps)))
        means = overlaps[draws].mean(axis=1)
        return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    m = len(a)
    size = max(k + 1, min(m - 1, int(round(0.8 * m))))
    full = float(fn(a, b))
    vals = np.array([fn(a[ idx := rng.choice(m, size=size, replace=False) ], b[idx]) for _ in range(n)], dtype=np.float64)
    se = float(vals.std(ddof=1)) * np.sqrt(size / (m - size))
    return float(full - 1.96 * se), float(full + 1.96 * se)


def subsample_ci(a, b, k=10, n=200, size=800, seed=0, z=1.96):
    """Normal-approximation 95% interval from n subsets of size rows without replacement."""
    a, b = np.asarray(a), np.asarray(b)
    m = len(a)
    if not 0 < size < m:
        raise ValueError(f"size must be between 1 and {m - 1}")
    rng = np.random.default_rng(seed)
    full = mutual_knn(a, b, k)
    sub = np.array([mutual_knn(a[idx := rng.choice(m, size=size, replace=False)], b[idx], k) for _ in range(n)])
    se = float(sub.std(ddof=1)) * np.sqrt(size / (m - size))
    return float(full - z * se), float(full + z * se)
