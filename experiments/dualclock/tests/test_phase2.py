import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.dualclock.fit_phase2_predictor import (
    fit_phase2_predictor,
)
from experiments.dualclock.phase1 import CollectionOptions, collect_c2i_batch
from experiments.dualclock.phase2 import (
    CAUSAL_FEATURES,
    CausalErrorPredictor,
    RefreshPolicy,
    RidgeHead,
    TerminalPiTCacheModel,
    paired_quality,
    sample_c2i_cached,
)
from pixdit_core.pixeldit_c2i import PixDiT


class Phase2Test(unittest.TestCase):
    @staticmethod
    def tiny_model() -> PixDiT:
        torch.manual_seed(53)
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

    @staticmethod
    def always_safe_predictor() -> CausalErrorPredictor:
        head = RidgeHead(
            mean=(0.0,) * len(CAUSAL_FEATURES),
            scale=(1.0,) * len(CAUSAL_FEATURES),
            coefficients=(-1.0,) + (0.0,) * len(CAUSAL_FEATURES),
        )
        return CausalErrorPredictor(
            CAUSAL_FEATURES,
            {1: head, 2: head, 3: head},
            {1: 0.0, 2: 0.0, 3: 0.0},
        )

    def test_exact_terminal_wrapper_matches_released_forward(self):
        model = self.tiny_model()
        wrapper = TerminalPiTCacheModel(model)
        x = torch.randn(4, 3, 4, 4)
        t = torch.linspace(0.1, 0.7, 4)
        y = torch.tensor([10, 10, 1, 2])
        expected = model(x, t, y)
        actual, state = wrapper.forward_exact(x, t, y)
        reinjected = wrapper.forward_cached(state)
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(torch.equal(actual, reinjected))

    def test_cached_path_skips_all_upstream_blocks(self):
        model = self.tiny_model()
        wrapper = TerminalPiTCacheModel(model)
        x = torch.randn(2, 3, 4, 4)
        t = torch.tensor([0.2, 0.2])
        y = torch.tensor([10, 3])
        _, state = wrapper.forward_exact(x, t, y)
        calls = {"patch": 0, "pit": 0, "final": 0}

        def count(name):
            def hook(*_args):
                calls[name] += 1

            return hook

        handles = [
            model.patch_blocks[0].register_forward_hook(count("patch")),
            model.pixel_blocks[0].register_forward_hook(count("pit")),
            model.final_layer.register_forward_hook(count("final")),
        ]
        try:
            wrapper.forward_cached(state)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(calls, {"patch": 0, "pit": 0, "final": 1})

    def test_exact_sampler_matches_phase1_ab2_trajectory(self):
        model = self.tiny_model()
        wrapper = TerminalPiTCacheModel(model)
        noise = torch.randn(1, 3, 4, 4)
        condition = torch.tensor([2])
        uncondition = torch.tensor([10])
        result = sample_c2i_cached(
            wrapper,
            noise.clone(),
            condition,
            uncondition,
            RefreshPolicy("exact"),
            num_steps=5,
        )
        trace = collect_c2i_batch(
            model,
            noise,
            condition,
            uncondition,
            CollectionOptions(
                num_steps=5,
                stale_horizons=(),
                activation_storage="sketch",
                sketch_dim=8,
                output_dtype=torch.float32,
                token_groups=0,
                channel_groups=0,
            ),
        )
        self.assertTrue(
            torch.allclose(
                result.sample,
                trace["final_sample"],
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertEqual(result.refresh_count, 5)
        self.assertEqual(result.cached_count, 0)

    def test_fixed_and_adaptive_refresh_decisions_are_guarded(self):
        model = self.tiny_model()
        wrapper = TerminalPiTCacheModel(model)
        noise = torch.randn(1, 3, 4, 4)
        condition = torch.tensor([1])
        uncondition = torch.tensor([10])
        fixed = sample_c2i_cached(
            wrapper,
            noise.clone(),
            condition,
            uncondition,
            RefreshPolicy("fixed", fixed_interval=2),
            num_steps=7,
        )
        self.assertEqual(
            [row["step"] for row in fixed.decisions if row["refresh"]],
            [0, 2, 4, 6],
        )

        adaptive = sample_c2i_cached(
            wrapper,
            noise,
            condition,
            uncondition,
            RefreshPolicy(
                "adaptive",
                predictor=self.always_safe_predictor(),
                forced_refresh_steps=frozenset({4}),
            ),
            num_steps=8,
        )
        refresh_steps = [
            row["step"] for row in adaptive.decisions if row["refresh"]
        ]
        self.assertEqual(refresh_steps[:4], [0, 1, 2, 4])
        self.assertEqual(
            adaptive.decisions[4]["reason"], "forced_guardrail"
        )
        self.assertLessEqual(
            max(row["cache_horizon"] for row in adaptive.decisions), 3
        )

    def test_predictor_export_and_paired_metrics(self):
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
                sketch_dim=8,
                output_dtype=torch.float32,
                token_groups=0,
                channel_groups=0,
            ),
        )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            shard = "trajectory.pt"
            torch.save(trace, directory / shard)
            with open(
                directory / "manifest.json", "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {
                        "schema_version": 1,
                        "shards": [
                            {
                                "file": shard,
                                "start": 0,
                                "count": batch_size,
                            }
                        ],
                    },
                    handle,
                )
            payload = fit_phase2_predictor(directory)
            predictor_path = directory / "predictor.json"
            with open(predictor_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, allow_nan=False)
            predictor = CausalErrorPredictor.from_file(
                predictor_path, "full", 0.75
            )
        self.assertEqual(set(payload["models"]), {"full", "timestep_only"})
        self.assertEqual(predictor.max_horizon, 3)
        self.assertEqual(tuple(predictor.feature_names), CAUSAL_FEATURES)

        exact = torch.randn(4, 3, 8, 8)
        metrics = paired_quality(exact, exact.clone())
        self.assertEqual(metrics["pixel_rmse"]["max"], 0.0)
        self.assertEqual(metrics["high_frequency_rmse"]["max"], 0.0)


if __name__ == "__main__":
    unittest.main()
