import tempfile
import unittest
from pathlib import Path

import torch

from experiments.dualclock.benchmark import account_component_envelopes, estimate_flops
from experiments.dualclock.common import load_weights
from experiments.dualclock.parity import compare, extract_and_reinject
from experiments.dualclock.run_phase0 import CASES
from experiments.dualclock.validate import trial_median_cv_percent
from pixdit_core.pixeldit_c2i import PixDiT
from pixdit_core.pixeldit_t2i import PixDiT_T2I


class Phase0Test(unittest.TestCase):
    @staticmethod
    def randomize(model):
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.is_floating_point():
                    parameter.normal_(mean=0.0, std=0.05)
        return model

    def test_c2i_semantic_reinjection_is_exact(self):
        torch.manual_seed(7)
        model = self.randomize(PixDiT(
            in_channels=3,
            num_groups=2,
            hidden_size=32,
            pixel_hidden_size=8,
            patch_depth=2,
            pixel_depth=1,
            patch_size=4,
            num_classes=10,
        )).eval()
        x = torch.randn(2, 3, 8, 8)
        t = torch.tensor([0.25, 0.75])
        y = torch.tensor([1, 2])
        exact, reinjected, semantic = extract_and_reinject(model, x, t, y)
        self.assertEqual(tuple(semantic.shape), (2, 4, 32))
        self.assertTrue(torch.equal(exact, reinjected))

    def test_t2i_semantic_reinjection_is_exact(self):
        torch.manual_seed(11)
        model = self.randomize(PixDiT_T2I(
            in_channels=3,
            num_groups=2,
            hidden_size=32,
            pixel_hidden_size=8,
            pixel_attn_hidden_size=32,
            pixel_num_groups=2,
            patch_depth=2,
            pixel_depth=1,
            patch_size=4,
            txt_embed_dim=24,
            txt_max_length=6,
        )).eval()
        x = torch.randn(1, 3, 8, 8)
        t = torch.tensor([0.5])
        y = torch.randn(1, 6, 24)
        exact, reinjected, _ = extract_and_reinject(model, x, t, y)
        self.assertTrue(torch.equal(exact, reinjected))

    def test_checkpoint_prefix_selection_prefers_ema(self):
        model = PixDiT(
            in_channels=3,
            num_groups=2,
            hidden_size=32,
            pixel_hidden_size=8,
            patch_depth=1,
            pixel_depth=1,
            patch_size=4,
            num_classes=10,
        )
        expected = {key: torch.full_like(value, 2) for key, value in model.state_dict().items()}
        checkpoint = {"state_dict": {f"ema_denoiser.{key}": value for key, value in expected.items()}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            torch.save(checkpoint, path)
            report = load_weights(model, str(path), "c2i")
        self.assertEqual(report["coverage"], 1.0)
        self.assertTrue(all(torch.equal(value, expected[key]) for key, value in model.state_dict().items()))

    def test_flop_breakdown_has_required_categories(self):
        model = PixDiT(
            in_channels=3,
            num_groups=2,
            hidden_size=32,
            pixel_hidden_size=8,
            patch_depth=1,
            pixel_depth=1,
            patch_size=4,
            num_classes=10,
        )
        result = estimate_flops(model, "c2i", 1, 8, 8, 0)
        required = {
            "patch_embedding_text_projection",
            "patch_attention",
            "patch_mlp",
            "patch_adaln",
            "pit_adaln",
            "pit_compaction_expansion",
            "pit_attention",
            "pit_mlp",
            "final_projection_reconstruction",
        }
        self.assertTrue(required.issubset(result))
        self.assertTrue(all(value >= 0 for value in result.values()))

    def test_component_envelopes_produce_exclusive_accounting(self):
        accounted, diagnostics = account_component_envelopes(
            {
                "__model_total": 100.0,
                "__patch_blocks_total": 60.0,
                "__pit_blocks_total": 20.0,
                "patch_attention": 20.0,
                "patch_mlp": 25.0,
                "patch_adaln": 5.0,
                "pit_attention": 5.0,
                "pit_mlp": 4.0,
                "pit_adaln": 3.0,
                "pit_compaction_expansion": 3.0,
                "conditioning": 5.0,
                "pixel_embedding": 2.0,
                "final_projection_reconstruction": 3.0,
            }
        )
        self.assertEqual(accounted["patch_residual_tensor_ops"], 10.0)
        self.assertEqual(accounted["pit_residual_tensor_ops"], 5.0)
        self.assertEqual(accounted["model_tensor_rearrangement"], 10.0)
        self.assertEqual(sum(accounted.values()), 100.0)
        self.assertEqual(diagnostics["instrumented_model_envelope_ms"], 100.0)

    def test_compare_detects_difference(self):
        result = compare(torch.zeros(4), torch.ones(4), atol=0.0, rtol=0.0)
        self.assertFalse(result["allclose"])
        self.assertEqual(result["max_abs"], 1.0)

    def test_t2i512_uses_released_stage3_recipe(self):
        case = CASES["t2i512"]
        self.assertIn("1024px_pixel_diffusion_stage3", case["config"])
        self.assertEqual(case["sampling"]["cfg_scale"], 2.75)
        self.assertEqual(case["sampling"]["flow_shift"], 4.0)

    def test_trial_median_cv_uses_sample_standard_deviation(self):
        latency = {"trial_medians_ms": [84.99011993408203, 82.08711242675781, 82.21331787109375]}
        self.assertAlmostEqual(trial_median_cv_percent(latency), 1.9746, places=3)


if __name__ == "__main__":
    unittest.main()
