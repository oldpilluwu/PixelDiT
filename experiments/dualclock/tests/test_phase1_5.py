import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.dualclock.phase1 import CollectionOptions, collect_c2i_batch
from experiments.dualclock.validate_alternative_directions import (
    CAUSAL_FEATURES,
    _bootstrap_mean_interval,
    _previous_velocity_features,
    validate_phase1_5,
)
from pixdit_core.pixeldit_c2i import PixDiT


class Phase15Test(unittest.TestCase):
    @staticmethod
    def tiny_model() -> PixDiT:
        torch.manual_seed(41)
        model = PixDiT(
            in_channels=3,
            num_groups=2,
            hidden_size=32,
            pixel_hidden_size=8,
            patch_depth=3,
            pixel_depth=2,
            patch_size=2,
            num_classes=10,
        )
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.is_floating_point():
                    parameter.normal_(mean=0.0, std=0.03)
        return model.eval()

    def test_previous_velocity_features_are_strictly_lagged(self):
        velocity = torch.tensor([0.5, 1.0, 2.0, 4.0, 7.0]).view(5, 1, 1)
        previous_change, previous_curvature = _previous_velocity_features(
            velocity
        )
        self.assertTrue(torch.isnan(previous_change[0, 0]))
        self.assertTrue(torch.isnan(previous_change[1, 0]))
        self.assertAlmostEqual(float(previous_change[2, 0]), 0.5)
        self.assertTrue(torch.isnan(previous_curvature[2, 0]))
        self.assertTrue(torch.isfinite(previous_curvature[3, 0]))

    def test_bootstrap_interval_preserves_clear_sign(self):
        lower, upper = _bootstrap_mean_interval(
            torch.tensor([0.1, 0.2, 0.3, 0.4]),
            samples=500,
            confidence=0.95,
            seed=11,
        )
        self.assertGreater(lower, 0.0)
        self.assertGreater(upper, lower)

    def test_end_to_end_phase1_5_report_uses_only_causal_features(self):
        model = self.tiny_model()
        batch_size = 8
        trace = collect_c2i_batch(
            model,
            torch.randn(batch_size, 3, 4, 4),
            torch.arange(batch_size) % 10,
            torch.full((batch_size,), 10),
            CollectionOptions(
                num_steps=8,
                stale_horizons=(1, 2, 3),
                activation_storage="sketch",
                sketch_dim=16,
                output_dtype=torch.float32,
                sensitivity_steps=(),
                token_groups=0,
                channel_groups=0,
            ),
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            shard = "trajectory_00000_00007.pt"
            torch.save(trace, directory / shard)
            with open(directory / "manifest.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": 1,
                        "shards": [
                            {"file": shard, "start": 0, "count": batch_size}
                        ],
                    },
                    handle,
                )
            report = validate_phase1_5(
                directory,
                folds=4,
                bootstrap_samples=200,
                seed=43,
            )

        self.assertEqual(report["trajectory_count"], batch_size)
        self.assertEqual(report["causal_features"], list(CAUSAL_FEATURES))
        self.assertNotIn("semantic_change", report["causal_features"])
        self.assertNotIn("raw_patch_change", report["causal_features"])
        self.assertEqual(len(report["held_out_predictors"]), 3)
        h1 = report["held_out_predictors"][
            "stale_generic/h1/guided/relative_rmse"
        ]
        self.assertEqual(h1["folds"], 4)
        self.assertGreater(h1["observation_count"], 0)
        self.assertIn("h1", report["paired_bootstrap"])
        self.assertEqual(
            set(report["policy_curves"]["fixed_intervals"]),
            {"interval_2", "interval_3", "interval_4"},
        )
        self.assertEqual(len(report["policy_curves"]["adaptive"]), 6)
        self.assertIn(
            report["recommendation"],
            {"run_small_pit_layer_sweep", "stop_or_refine_predictor"},
        )
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
