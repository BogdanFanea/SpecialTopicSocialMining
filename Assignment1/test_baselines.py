"""Small offline checks for label alignment, AUROC, and reproducible probes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from baselines import aligned_test_labels, check_features, fit_probe, metrics, score_checkpoint


class ProtocolChecks(unittest.TestCase):
    def test_test_labels_join_by_id_and_reject_bad_inputs(self):
        rows = [{"id": i, "text": f"sample {i}", "img": f"img/{i}.png"} for i in [2, 1]]
        annotations = [{**r, "label": r["id"] % 2} for r in reversed(rows)]
        self.assertEqual(aligned_test_labels(rows, annotations), {1: 1, 2: 0})
        for bad in [annotations[:1], annotations + annotations[:1],
                    [{**annotations[0], "text": "changed"}, annotations[1]],
                    [{**annotations[0], "label": -1}, annotations[1]]]:
            with self.assertRaises(ValueError):
                aligned_test_labels(rows, bad)

    def test_continuous_auroc_direction_and_ties(self):
        self.assertEqual(metrics([0, 1], [.1, .9])["auroc"], 1)
        self.assertEqual(metrics([0, 1], [.9, .1])["auroc"], 0)
        self.assertEqual(metrics([0, 1], [.5, .5])["auroc"], .5)
        # All four probabilities produce the same thresholded class, but distinct rankings.
        self.assertEqual(metrics([0, 0, 1, 1], [.1, .2, .3, .4])["auroc"], 1)
        with self.assertRaises(ValueError):
            metrics([0, 1], [.1, float("nan")])

    def test_cache_alignment_and_finite_values(self):
        x = np.ones((2, 3), dtype=np.float32)
        check_features(x, [3, 4], [3, 4], 3)
        with self.assertRaises(ValueError):
            check_features(x, [4, 3], [3, 4], 3)
        with self.assertRaises(ValueError):
            check_features(x, [3, 4], [3, 4], 4)
        x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            check_features(x, [3, 4], [3, 4], 3)

    def test_training_only_scaling_reproducibility_and_reload(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(7)
        x = rng.normal(size=(40, 4)).astype(np.float32)
        v = (rng.normal(size=(12, 4)) + .3).astype(np.float32)
        y = torch.tensor((x[:, 0] > 0).astype(np.float32))
        yv = (v[:, 0] > .3).astype(int)
        scaler = StandardScaler().fit(x)
        mean = torch.from_numpy(scaler.mean_.astype(np.float32))
        scale = torch.from_numpy(scaler.scale_.astype(np.float32))
        np.testing.assert_allclose(mean, x.mean(0), atol=1e-7)
        self.assertFalse(np.allclose(mean, np.concatenate([x, v]).mean(0)))
        xt, xv = (torch.from_numpy(x)-mean)/scale, (torch.from_numpy(v)-mean)/scale
        kwargs = dict(seed=42, lr=.01, weight_decay=.01, batch_size=8, epochs=5)
        a, history_a = fit_probe(xt, y, xv, yv, **kwargs)
        b, history_b = fit_probe(xt, y, xv, yv, **kwargs)
        self.assertEqual(history_a, history_b)
        for k in a["state_dict"]:
            torch.testing.assert_close(a["state_dict"][k], b["state_dict"][k], rtol=0, atol=0)
        checkpoint = {**a, "mean": mean, "scale": scale}
        scores = score_checkpoint(checkpoint, v)
        self.assertEqual(metrics(yv, scores)["auroc"], a["validation_auroc"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.pt"
            torch.save(checkpoint, path)
            loaded = torch.load(path, weights_only=True)
            np.testing.assert_array_equal(scores, score_checkpoint(loaded, v))
        self.assertEqual(set(a["state_dict"]), {"weight", "bias"})
        self.assertEqual(a["state_dict"]["weight"].shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
