import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.dualclock.analyze_alternative_directions import (
    _auc,
    _residual_rank,
    analyze_alternatives,
)
from experiments.dualclock.phase1 import CollectionOptions, collect_c2i_batch
from pixdit_core.pixeldit_c2i import PixDiT


class AlternativeDirectionsTest(unittest.TestCase):
    @staticmethod
    def tiny_model() -> PixDiT:
        torch.manual_seed(31)
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

    def test_auc_handles_ties_and_direction(self):
        labels = torch.tensor([False, True, False, True])
        self.assertAlmostEqual(_auc(torch.ones(4), labels), 0.5)
        self.assertAlmostEqual(
            _auc(torch.tensor([0.0, 3.0, 1.0, 2.0]), labels), 1.0
        )
        self.assertAlmostEqual(
            _auc(torch.tensor([3.0, 0.0, 2.0, 1.0]), labels), 0.0
        )

    def test_residual_rank_detects_low_dimensional_motion(self):
        torch.manual_seed(37)
        time = torch.linspace(-1.0, 1.0, 20)
        basis = torch.randn(1, 32)
        low_rank = (time.square().view(-1, 1, 1) * basis).contiguous()
        high_rank = torch.randn(20, 1, 32)
        low_report = _residual_rank(low_rank)
        high_report = _residual_rank(high_rank)
        self.assertLessEqual(float(low_report["rank_95"][0]), 1.0)
        self.assertGreater(
            float(high_report["rank_95"][0]),
            float(low_report["rank_95"][0]),
        )

    def test_tiny_trace_populates_alternative_direction_data(self):
        model = self.tiny_model()
        options = CollectionOptions(
            num_steps=6,
            stale_horizons=(1, 2),
            activation_storage="sketch",
            sketch_dim=16,
            output_dtype=torch.float32,
            sensitivity_steps=(2,),
            token_groups=2,
            channel_groups=2,
        )
        trace = collect_c2i_batch(
            model,
            torch.randn(2, 3, 4, 4),
            torch.tensor([2, 7]),
            torch.tensor([10, 10]),
            options,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            shard_name = "trajectory_00000_00001.pt"
            torch.save(trace, directory / shard_name)
            with open(directory / "manifest.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": 1,
                        "shards": [
                            {"file": shard_name, "start": 0, "count": 2}
                        ],
                    },
                    handle,
                )

            report = analyze_alternatives(
                directory,
                analysis_dim=16,
                harmful_threshold=1e-6,
                top_k=4,
            )

        self.assertEqual(report["mode"], "c2i")
        self.assertEqual(report["sample_count"], 2)
        self.assertTrue(
            any(
                item["name"].startswith("pit/block_")
                for item in report["layer_rankings"]
            )
        )
        self.assertIn(
            "stale_generic/h1/guided/relative_rmse",
            report["region_cache_errors"],
        )
        self.assertGreater(
            report["raw_patch_residual_rank"]["rank_95"]["count"], 0
        )
        self.assertGreater(
            report["cfg_redundancy"][
                "raw_patch_delta_to_shared_norm"
            ]["count"],
            0,
        )
        self.assertGreater(
            report["raw_patch_volatility"]["observation_count"], 0
        )
        self.assertTrue(report["raw_patch_volatility"]["top_tokens"])
        self.assertTrue(report["raw_patch_volatility"]["top_channels"])
        self.assertTrue(report["decoder_sensitivity_rankings"])
        self.assertTrue(
            report["velocity_dynamics"]["most_frequently_hard_steps"]
        )
        self.assertIn("velocity_high_to_low", report["frequency_evolution"])
        self.assertIn(
            "stale_raw_patch/h1/guided/relative_rmse",
            report["error_predictors"],
        )
        self.assertIn(
            "stale_generic/h1/guided/relative_rmse",
            report["error_by_content_group"],
        )
        self.assertEqual(len(report["directions"]), 5)
        # The CLI writer consumes this object directly.
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
