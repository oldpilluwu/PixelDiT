import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.dualclock.analyze_temporal_dynamics import (
    analyze,
    predicted_speedup,
    temporal_measurements,
)
from experiments.dualclock.phase1 import (
    CollectionOptions,
    collect_c2i_batch,
    guided_velocity,
    shifted_timesteps,
)
from experiments.dualclock.phase1_t2i import collect_t2i_batch
from pixdit_core.pixeldit_c2i import PixDiT
from pixdit_core.pixeldit_t2i import PixDiT_T2I


class Phase1Test(unittest.TestCase):
    @staticmethod
    def tiny_model() -> PixDiT:
        torch.manual_seed(17)
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

    def test_temporal_measurements_distinguish_slow_and_fast_states(self):
        time = torch.linspace(0, 1, 12)
        slow = torch.stack((time, time.square()), dim=-1).view(12, 1, 2)
        fast = torch.stack((torch.sin(25 * time), torch.cos(19 * time)), dim=-1).view(12, 1, 2)
        slow_curvature = temporal_measurements(slow)["normalized_curvature"].median()
        fast_curvature = temporal_measurements(fast)["normalized_curvature"].median()
        self.assertLess(slow_curvature, fast_curvature)
        self.assertIn("cosine_lag_5", temporal_measurements(slow))

    def test_collection_records_required_exact_and_hooked_states(self):
        model = self.tiny_model()
        options = CollectionOptions(
            num_steps=4,
            stale_horizons=(1, 2),
            activation_storage="sketch",
            sketch_dim=16,
            output_dtype=torch.float32,
            sensitivity_steps=(1,),
            token_groups=2,
            channel_groups=2,
        )
        trace = collect_c2i_batch(
            model,
            torch.randn(1, 3, 4, 4),
            torch.tensor([2]),
            torch.tensor([10]),
            options,
        )
        self.assertEqual(tuple(trace["exact"]["x_t"].shape), (4, 1, 3, 4, 4))
        self.assertEqual(tuple(trace["exact"]["semantic"].shape), (4, 2, 4, 32))
        self.assertEqual(
            tuple(trace["exact"]["raw_final_patch"].shape), (4, 2, 4, 32)
        )
        self.assertEqual(tuple(trace["exact"]["velocity_branches"].shape), (4, 2, 3, 4, 4))
        self.assertIn("patch/block_0", trace["representations"])
        self.assertIn("pit/block_0/input", trace["representations"])
        self.assertIn("pit/block_1/output", trace["representations"])
        self.assertIn("patch/block_0/head_q", trace["representations"])
        self.assertIn("stale_semantic/h2/guided/relative_rmse", trace["substitutions"])
        self.assertIn(
            "stale_raw_patch/h2/guided/relative_rmse",
            trace["substitutions"],
        )
        self.assertIn(
            "linear_raw_patch/h2/guided/relative_rmse",
            trace["substitutions"],
        )
        self.assertIn("stale_generic/h1/guided/relative_rmse", trace["substitutions"])
        self.assertEqual(len(trace["sensitivity"]), 4)

    def test_collector_trajectory_uses_exact_ab2_velocity(self):
        model = self.tiny_model()
        noise = torch.randn(1, 3, 4, 4)
        condition = torch.tensor([1])
        uncondition = torch.tensor([10])
        options = CollectionOptions(
            num_steps=4,
            stale_horizons=(),
            activation_storage="sketch",
            sketch_dim=8,
            output_dtype=torch.float32,
            token_groups=0,
            channel_groups=0,
        )
        trace = collect_c2i_batch(model, noise.clone(), condition, uncondition, options)
        steps = shifted_timesteps(4, 1.0, noise.device, noise.dtype)
        x = noise
        previous = None
        cfg_condition = torch.cat((uncondition, condition))
        with torch.inference_mode():
            for t_cur, t_next in zip(steps[:-1], steps[1:]):
                timestep = t_cur.repeat(1)
                branches = model(
                    torch.cat((x, x)), timestep.repeat(2), cfg_condition
                )
                velocity = guided_velocity(branches, timestep, 2.75, 0.1, 0.9)
                dt = t_next - t_cur
                x = x + velocity * dt if previous is None else x + dt * (1.5 * velocity - 0.5 * previous)
                previous = velocity
        self.assertTrue(torch.allclose(trace["final_sample"], x, atol=1e-6, rtol=1e-6))

    def test_t2i_collection_uses_official_solver_and_joint_hooks(self):
        torch.manual_seed(23)
        model = PixDiT_T2I(
            in_channels=3,
            num_groups=2,
            hidden_size=32,
            pixel_hidden_size=8,
            pixel_attn_hidden_size=32,
            pixel_num_groups=2,
            patch_depth=2,
            pixel_depth=1,
            patch_size=2,
            txt_embed_dim=24,
            txt_max_length=6,
        ).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.is_floating_point():
                    parameter.normal_(mean=0.0, std=0.03)
        options = CollectionOptions(
            num_steps=3,
            cfg_scale=2.75,
            guidance_min=0.0,
            guidance_max=1.0,
            stale_horizons=(1, 2),
            activation_storage="sketch",
            sketch_dim=16,
            output_dtype=torch.float32,
            token_groups=0,
            channel_groups=0,
        )
        trace = collect_t2i_batch(
            model,
            torch.randn(1, 3, 4, 4),
            torch.randn(1, 6, 24),
            torch.randn(1, 6, 24),
            options,
            flow_shift=4.0,
        )
        self.assertEqual(trace["mode"], "t2i")
        self.assertEqual(tuple(trace["exact"]["timestep"].shape), (3, 1))
        self.assertEqual(tuple(trace["exact"]["semantic"].shape), (3, 2, 4, 32))
        self.assertIn("text/block_0", trace["representations"])
        self.assertIn("text/block_1/head_q", trace["representations"])
        self.assertIn(
            "linear_raw_patch/h2/guided/relative_rmse",
            trace["substitutions"],
        )

    def test_analysis_writes_all_phase1_sections(self):
        model = self.tiny_model()
        options = CollectionOptions(
            num_steps=5,
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
            torch.randn(1, 3, 4, 4),
            torch.tensor([3]),
            torch.tensor([10]),
            options,
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            torch.save(trace, directory / "trajectory_00000_00000.pt")
            with open(directory / "manifest.json", "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": 1,
                        "shards": [
                            {
                                "file": "trajectory_00000_00000.pt",
                                "start": 0,
                                "count": 1,
                            }
                        ],
                    },
                    handle,
                )
            baseline = {
                "results": [
                    {
                        "batch_size": 1,
                        "component_envelopes_ms": {
                            "instrumented_model_envelope_ms": {"median": 10.0}
                        },
                        "components_ms": {
                            "conditioning": {"median": 0.2},
                            "patch_embedding_text_projection": {"median": 0.2},
                            "patch_attention": {"median": 3.0},
                            "patch_mlp": {"median": 2.0},
                            "patch_adaln": {"median": 0.5},
                            "patch_residual_tensor_ops": {"median": 0.5},
                            "pit_attention": {"median": 1.0},
                        },
                    }
                ]
            }
            with open(directory / "baseline.json", "w", encoding="utf-8") as handle:
                json.dump(baseline, handle)
            report = analyze(directory, str(directory / "baseline.json"))
        self.assertIn("regions", report["representations"]["final_semantic"])
        self.assertIn("semantic_curvature_by_cfg_branch", report)
        self.assertIn("semantic_curvature_by_image_content", report)
        self.assertIn("semantic_velocity_error_correlation", report)
        self.assertIn("decoder_sensitivity", report)
        self.assertIn("candidates", report["gate"])
        self.assertEqual(report["gate"]["status"], "insufficient_evidence")
        self.assertFalse(
            any("/head_" in name for name in report["gate"]["candidates"])
        )
        self.assertIn("raw_final_patch", report["gate"]["candidates"])
        raw_curvature = report["representations"]["raw_final_patch"]["overall"][
            "normalized_curvature"
        ]["median"]
        hooked_curvature = report["representations"]["patch/block_2"]["overall"][
            "normalized_curvature"
        ]["median"]
        self.assertAlmostEqual(raw_curvature, hooked_curvature, places=6)

    def test_cost_model_speedup(self):
        cost = {"available": True, "semantic_fraction": 0.75}
        self.assertAlmostEqual(predicted_speedup(cost, 2), 1.6)


if __name__ == "__main__":
    unittest.main()
