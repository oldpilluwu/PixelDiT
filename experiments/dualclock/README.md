# DualClock experiments

This directory implements Phase 0 baseline reproduction and Phase 1
slow-semantics experiments without changing `pixdit_core/`. Reports, generated
images, traces, and checkpoints are ignored by Git.

## A6000 setup

The released repository recommends NVIDIA's PyTorch 24.09 image. On the server:

```bash
git clone <your-pixeldit-repository> PixelDiT
cd PixelDiT

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/PixelDiT -w /workspace/PixelDiT -it \
  nvcr.io/nvidia/pytorch:24.09-py3

pip install -r requirements.txt
python -m experiments.dualclock.tests.test_phase0
nvidia-smi
```

For a native environment, use Python 3.10 or 3.11 and a CUDA-enabled PyTorch
build instead of the Docker command. The profiler refuses to run without CUDA.
An RTX A6000 is Ampere (SM 8.6) and supports the default BF16 measurements.

## One-command instrumentation

Start with C2I 256, as required by the plan:

```bash
python -m experiments.dualclock.run_phase0 \
  --cases c2i256 \
  --dtype bf16 \
  --batch-sizes 1,auto \
  --max-auto-batch 32 \
  --warmup 10 --repeats 20 --trials 3
```

This downloads the released epoch-320 checkpoint when needed and writes a
timestamped directory under `experiments/dualclock/reports/` containing:

- Git, dependency, CUDA, cuDNN, driver, GPU clock, power, memory, and checkpoint
  hash metadata.
- Exact semantic extraction/reinjection and deterministic-repeat checks.
- CUDA-event p10/median/p90 latency, peak allocated memory, throughput,
  component timing, analytical component FLOPs, timing coverage, and
  run-to-run variation for batch 1 and the largest batch that fits.
- Fixed 1,000-sample C2I, smoke, and 100-prompt T2I manifests.
- An `instrumentation_gate.json` summary. The command exits nonzero if semantic
  parity, <3% timing variation, or >=95% component timing coverage is not met;
  all reports remain on disk for diagnosis.

After C2I 256 passes, run later baselines in order:

```bash
python -m experiments.dualclock.run_phase0 --cases c2i512 \
  --batch-sizes 1,auto --max-auto-batch 8

python -m experiments.dualclock.run_phase0 --cases t2i512 \
  --batch-sizes 1,auto --max-auto-batch 8

python -m experiments.dualclock.run_phase0 --cases t2i1024 \
  --batch-sizes 1,auto --max-auto-batch 2
```

To compare eager and `torch.compile`, add `--compiled`. Component hooks are
used only for eager profiling; compiled runs are deliberately total-only so
instrumentation hooks do not introduce graph breaks:

```bash
python -m experiments.dualclock.run_phase0 --cases c2i256 \
  --batch-sizes 1 --compiled --compile-mode default
```

The automatic batch search intentionally allocates up to the given cap. Reduce
the cap on a shared GPU. Run with no other GPU workload, lock clocks if your
server policy permits it, and record any clock lock in the environment report.
Clean end-to-end latency is collected without module hooks. Component timings
are collected in a separate instrumented pass (`--component-repeats`, default
10), and the report records that pass's overhead relative to the clean median.

## Generate frozen C2I regression outputs

Create the deterministic manifests:

```bash
python -m experiments.dualclock.make_regression_set \
  --output-dir experiments/dualclock/regression
```

Generate the rapid 1,000-image C2I-256 baseline (one class/seed pair per row):

```bash
python -m experiments.dualclock.sample_c2i \
  --config c2i/configs/pix256_xl.yaml \
  --checkpoint imagenet256_pixeldit_xl_epoch320.ckpt \
  --manifest experiments/dualclock/regression/c2i_1000.jsonl \
  --output-dir experiments/dualclock/reports/c2i256_samples \
  --height 256 --width 256 --batch-size 8 \
  --num-steps 100 --cfg-scale 2.75 --timeshift 1.0 \
  --guidance-min 0.1 --guidance-max 0.9
```

Generate the C2I-512 counterpart:

```bash
python -m experiments.dualclock.sample_c2i \
  --config c2i/configs/pix512_xl.yaml \
  --checkpoint imagenet512_pixeldit_xl.ckpt \
  --manifest experiments/dualclock/regression/c2i_1000.jsonl \
  --output-dir experiments/dualclock/reports/c2i512_samples \
  --height 512 --width 512 --batch-size 2 \
  --num-steps 100 --cfg-scale 3.5 --timeshift 2.0 \
  --guidance-min 0.1 --guidance-max 1.0
```

Use `c2i_smoke.jsonl` first with `--limit 25` for a short end-to-end check.
Every sample filename records its index, class, and seed, and `run.json` records
the solver/NFE/CFG/precision settings and checkpoint hash.

Freeze generated outputs into a hash manifest:

```bash
python -m experiments.dualclock.freeze_outputs \
  --input-dir experiments/dualclock/reports/c2i256_samples \
  --run-metadata experiments/dualclock/reports/c2i256_samples/run.json \
  --output experiments/dualclock/reports/c2i256_outputs.json
```

## Generate T2I regression outputs

The manifest generator writes exactly 100 prompts spanning texture, geometry,
people, animals, text rendering, repeated structures, entity counting, and
spatial relations. The published checkpoint is a 1024px stage-3 model; there is
no separately released 512px checkpoint. The 512 run therefore evaluates the
released stage-3 recipe at a lower output resolution and must be labeled as
such. Do not pair the released checkpoint with the stage-1 training config.

```bash
cd t2i
python inference.py \
  --config configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml \
  --model_path pixeldit_t2i_v1.pth \
  --txt_file ../experiments/dualclock/regression/t2i_prompts_100.txt \
  --custom_height 512 --custom_width 512 \
  --cfg_scale 2.75 --step 50 --seed 2045000 \
  --negative_prompt "low quality, worst quality, over-saturated, blurry, deformed, watermark" \
  --sample_nums 100 --bs 1 \
  --work_dir ../experiments/dualclock/reports/t2i512_samples

python inference.py \
  --config configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml \
  --model_path pixeldit_t2i_v1.pth \
  --txt_file ../experiments/dualclock/regression/t2i_prompts_100.txt \
  --custom_height 1024 --custom_width 1024 \
  --cfg_scale 2.75 --step 50 --seed 2045000 \
  --negative_prompt "low quality, worst quality, over-saturated, blurry, deformed, watermark" \
  --sample_nums 100 --bs 1 \
  --work_dir ../experiments/dualclock/reports/t2i1024_samples
cd ..
```

Run `freeze_outputs` on each generated image directory after inspection.

## Reading the reports

Phase 0 is ready to freeze only when:

- `semantic_reinjection.allclose` and `repeatability.allclose` are true.
- Trial-median sample coefficient of variation is below 3% after warmup.
  The max-minus-min range is retained separately as a diagnostic because it is
  too sensitive to serve as the gate with only three trials.
- `component_coverage_percent` is at least 95%; otherwise inspect
  `unattributed_ms` before drawing optimization conclusions.
- Output hashes reproduce on a clean rerun with the same checkpoint, seed,
  precision, and compile mode.

`parity.py --check-mask` records T2I mask acceptance separately. This is a
diagnostic, not a correctness fix: Phase 0 must preserve the released baseline,
and adjacent mask changes belong to a separately measured corrected baseline.

For a focused command, each tool is independently runnable:

```bash
python -m experiments.dualclock.capture_environment --help
python -m experiments.dualclock.parity --help
python -m experiments.dualclock.benchmark --help
python -m experiments.dualclock.sample_c2i --help
```

## Phase 1: collect exact trajectories

Phase 1 starts from the accepted official C2I-256 track. The collector verifies
semantic extraction/reinjection parity before writing any trace, follows the
released AB2/Flow-DPM sampling update, and keeps unconditional and conditional
CFG branches separate.

```bash
python -m experiments.dualclock.collect_trajectories \
  --config c2i/configs/pix256_xl.yaml \
  --checkpoint imagenet256_pixeldit_xl_epoch320.ckpt \
  --manifest experiments/dualclock/regression/c2i_1000.jsonl \
  --baseline-report experiments/dualclock/reports/20260717T042058Z/c2i256_eager.json \
  --height 256 --width 256 --limit 100 --batch-size 1 \
  --num-steps 100 --cfg-scale 2.75 --timeshift 1.0 \
  --guidance-min 0.1 --guidance-max 0.9
```

Each shard contains exact `x_t`, full final semantic tokens, conditional and
unconditional velocities, guided velocity, selected early/middle/late patch
states, every PiT block input/output, and per-head Q/K/V summaries. It also
contains same-state errors for stale semantics, linear semantic forecasts, and
equivalently stale generic PiT features; grouped token/channel interventions
measure decoder influence and empirical Lipschitz ratios.

The default `--activation-storage sketch` bounds the patch/PiT archive size.
These activations are deterministic signed pooled temporal sketches with exact L2
norms; exact `x_t`, final semantics, and velocities are still retained. Use
`--activation-storage full` for a small number of exact activation-heavy
trajectories. Full mode is intentionally explicit because all PiT inputs and
outputs at 100 evaluations consume substantial storage.

As in the released sampler, noise and the solver state remain FP32 while model
operations run under BF16 autocast. The default `--trace-dtype same` preserves
semantic activations and velocities in their native model dtype; solver states
are always archived in their native FP32 dtype.

Run analysis on the timestamped trace directory:

```bash
python -m experiments.dualclock.analyze_temporal_dynamics \
  --trace-dir experiments/dualclock/traces/<timestamp>
```

The analyzer reports normalized first and second differences, cosine
similarities at lags 1–5, temporal rank, timestep-frequency energy, early/mid/
late behavior, CFG branch and class splits, image-frequency splits, per-layer
and per-head variation, substitution errors, semantic/velocity error
correlation, decoder sensitivity, and a latency-based refresh cost model.
High-dimensional temporal calculations use a bounded deterministic signed projection
(`--analysis-dim`, default 4096), while the archived final semantics remain
full.

It writes `phase1_report.json`, `phase1_report.md`, and `phase1_gate.json`.
Thresholds are explicit CLI parameters. A `PASS` is evidence to begin Phase 2,
not a measured acceleration claim; a failed gate means the plan says to stop or
redirect DualClock.

For a smoke test, use `--limit 2 --num-steps 5`, disable the expensive grouped
influence probes with `--token-groups 0 --channel-groups 0`, and then analyze
the resulting directory. Smoke analysis is reported as `INSUFFICIENT EVIDENCE`;
the default hypothesis gate requires at least 100 trajectories with 100 solver
evaluations. Run the complete local test suite with:

```bash
python -m unittest \
  experiments.dualclock.tests.test_phase0 \
  experiments.dualclock.tests.test_phase1 -v
```
