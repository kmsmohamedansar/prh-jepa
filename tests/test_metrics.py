import pathlib, sys, unittest
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from metrics import mutual_knn, linear_cka, shuffled_baseline


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

    def test_rotation_invariant(self):
        self.assertAlmostEqual(mutual_knn(self.a, self.rot, 10), 1.0)
        self.assertAlmostEqual(linear_cka(self.a, self.rot), 1.0)

    def test_unrelated_near_chance(self):
        self.assertLess(mutual_knn(self.a, self.other, 10), 0.15)

    def test_shuffle_destroys_alignment(self):
        self.assertLess(shuffled_baseline(self.a, self.a, 10), 0.15)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            mutual_knn(self.a, self.a[:10])
        with self.assertRaises(ValueError):
            mutual_knn(self.a, self.a, k=200)


if __name__ == "__main__":
    unittest.main()
