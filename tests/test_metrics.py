import pathlib, sys, unittest
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from metrics import (
    bootstrap,
    linear_cka,
    mutual_knn,
    per_item_mutual_knn,
    prh_prep,
    prh_score,
    shuffled_baseline,
    subsample_ci,
    unbiased_cka,
)


class Metrics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.a = rng.normal(size=(200, 32))
        q, _ = np.linalg.qr(rng.normal(size=(32, 32)))
        self.rot = self.a @ q
        self.other = rng.normal(size=(200, 48))

    def test_identical_is_one(self):
        self.assertAlmostEqual(mutual_knn(self.a, self.a, 10), 1.0)
        self.assertAlmostEqual(linear_cka(self.a, self.a), 1.0)
        self.assertAlmostEqual(unbiased_cka(self.a, self.a), 1.0)

    def test_rotation_invariant(self):
        self.assertAlmostEqual(mutual_knn(self.a, self.rot, 10), 1.0)
        self.assertAlmostEqual(linear_cka(self.a, self.rot), 1.0)
        self.assertAlmostEqual(unbiased_cka(self.a, self.rot), 1.0)

    def test_unrelated_near_chance(self):
        self.assertLess(mutual_knn(self.a, self.other, 10), 0.15)
        self.assertLess(abs(unbiased_cka(self.a, self.other)), 0.1)

    def test_shuffle_destroys_alignment(self):
        self.assertLess(shuffled_baseline(self.a, self.a, 10), 0.15)

    def test_per_item_and_bootstrap_preserve_sample_size(self):
        noisy = self.rot + 0.5 * self.other[:, :32]
        point = mutual_knn(self.a, noisy, 10)
        items = per_item_mutual_knn(self.a, noisy, 10)
        self.assertEqual(len(items), len(self.a))
        self.assertAlmostEqual(float(items.mean()), point)
        lo, hi = bootstrap(self.a, noisy, k=10, n=500, seed=0)
        self.assertLess(lo, point)
        self.assertGreater(hi, point)
        sub_lo, sub_hi = subsample_ci(self.a, noisy, k=10, n=50, size=160, seed=0)
        self.assertLess(sub_lo, point)
        self.assertGreater(sub_hi, point)

    def test_prh_prep_and_prh_score(self):
        x = self.a.copy()
        x[0, 0] = 1000.0
        prep = prh_prep(x)
        norms = np.linalg.norm(prep, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(x)), atol=1e-6)
        best, i, j = prh_score([self.other, self.a], [self.other[:, :16], self.a], k=10, metric="mknn")
        self.assertAlmostEqual(best, 1.0)
        self.assertEqual((i, j), (1, 1))
        # Elementwise 0.95-quantile clipping depends on coordinate axes, so a general rotation drops below 1.0
        rot_best, ri, rj = prh_score([self.other, self.a], [self.other[:, :16], self.rot], k=10, metric="mknn")
        self.assertEqual((ri, rj), (1, 1))
        self.assertGreater(rot_best, 0.75)
        self.assertLess(rot_best, 1.0)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            mutual_knn(self.a, self.a[:10])
        with self.assertRaises(ValueError):
            mutual_knn(self.a, self.a, k=200)
        with self.assertRaises(ValueError):
            unbiased_cka(self.a[:3], self.a[:3])


if __name__ == "__main__":
    unittest.main()
