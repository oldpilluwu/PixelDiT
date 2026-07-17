<p align="center">
  <img src="assets/pixeldit-logo.png" height="120" />
</p>
 
<h2 align="center">PixelDiT: Pixel Diffusion Transformers for Image Generation</h2>

<p align="center">
  <a href="https://www.yongshengyu.com/">Yongsheng Yu</a><sup>1,2</sup> &nbsp;
  <a href="https://wxiong.me/">Wei Xiong</a><sup>1†</sup> &nbsp;
  <a href="https://weilinie.github.io/">Weili Nie</a><sup>1</sup> &nbsp;
  <a href="https://shengcn.github.io/">Yichen Sheng</a><sup>1</sup> &nbsp;
  <a href="http://behindthepixels.io/">Shiqiu Liu</a><sup>1</sup> &nbsp;
  <a href="https://www.cs.rochester.edu/u/jluo/">Jiebo Luo</a><sup>2</sup>
</p>
<p align="center">
  <sup>1</sup>NVIDIA &nbsp; <sup>2</sup>University of Rochester
  <br>
  <sup>†</sup>Project Lead and Main Advising
</p>

<p align="center">
  <a href="https://pixeldit.github.io/"><img src="https://img.shields.io/badge/%F0%9F%8C%90_-Project-2ea44f" /></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2511.20645"><img src="https://img.shields.io/badge/%F0%9F%93%84_-arXiv-b31b1b.svg" /></a>
  &nbsp;
  <a href="https://huggingface.co/nvidia/PixelDiT-ImageNet"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Model-ImageNet-yellow" /></a>
  &nbsp;
  <a href="https://huggingface.co/nvidia/PixelDiT-1300M-1024px"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Model-T2I-yellow" /></a> &nbsp; 
  <a href="https://paperswithcode.co/benchmark/imagenet-256x256?task=image-generation&eval=8057"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpaperswithcode.co%2Fapi%2Fv1%2Fevaluations%2F8057&query=%24.best_rank&prefix=Rank%20%23&label=ImageNet%20256%C3%97256%20FID&color=blue" /></a> <br>
  <a href="https://paperswithcode.co/conferences/cvpr-2026/best-paper-finalists"><img src="https://img.shields.io/badge/%F0%9F%8F%86_CVPR_2026-Best_Paper_Finalist-gold" /></a> &nbsp;
</p>

<p align="center">
  <img src="assets/pixeldit-t2i.jpg" width="100%" />
</p>

PixelDiT is a single-stage, end-to-end pixel-space diffusion transformer that eliminates the VAE autoencoder entirely. It uses a dual-level architecture — patch-level DiT for global semantics + pixel-level DiT for texture details — to generate images directly in pixel space.

- **1.61 FID** on ImageNet 256×256
- **0.74 GenEval** / **83.5 DPG-Bench** on text-to-image at 1024×1024
- No VAE, no latent space

## 🔥 News 

- **[2026/06]** Added a **post-modulation** option for the PiT (pixel-level) blocks that mitigates the training loss spikes ([#6](https://github.com/NVlabs/PixelDiT/issues/6)). See [c2i/README.md](c2i/README.md#training-stability-post-modulation-for-pit-blocks).
- **[2026/06]** PixelDiT is selected as a CVPR 2026 Best Paper Finalist.
- **[2026/04]** Training & inference code, and pre-trained models are released.
- **[2026/02]** PixelDiT is accepted to CVPR 2026 Oral. 
- **[2025/11]** [arxiv](https://arxiv.org/abs/2511.20645) is released.

## Performance

### ImageNet 256×256 (PixelDiT-XL, 797M params)

All evaluations use **FlowDPMSolver** with **100 steps**. 50K samples. Metrics follow ADM evaluation protocol.

| Epoch | gFID↓ | CFG Scale | Steps | Sampler | Time Shift | CFG Interval |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 80  | **2.36** | 3.25 | 100 | FlowDPMSolver | 1.0 | [0.1, 1.0] |
| 160 | **1.97**  | 3.25 | 100 | FlowDPMSolver | 1.0 | [0.1, 1.0] |
| 320 | **1.61** | 2.75 | 100 | FlowDPMSolver | 1.0 | [0.1, 0.9] |

### ImageNet 512×512 (PixelDiT-XL, 797M params)

| Resolution | gFID↓ | CFG Scale | Steps | Sampler | Time Shift | CFG Interval |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 512×512 | **1.81** | 3.5 | 100 | FlowDPMSolver | 2.0 | [0.1, 1.0] |

### Text-to-Image (PixelDiT-T2I, 1.3B params)

| Resolution | GenEval↑ | DPG-Bench↑ |
|:---:|:---:|:---:|
| 512×512  | 0.78 | 83.7 |
| 1024×1024 | 0.74 | 83.5 |

## Getting Started

**Docker image** (recommended): `nvcr.io/nvidia/pytorch:24.09-py3`

```bash
pip install -r requirements.txt
```

## Tasks

> **Note:** Our models are resumed every 4 hours, using the timestamp as the random seed each time. As a result, the final training outcome may have a slight gap compared to a continuous run without intermediate resumes.

### Class-to-Image Generation (ImageNet)

Training and evaluation instructions for class-conditioned generation on ImageNet 256×256 and 512×512.

→ **[c2i/README.md](c2i/README.md)**

### Text-to-Image Generation

Multi-stage training (512px → 1024px) and inference for text-to-image generation.

→ **[t2i/README.md](t2i/README.md)**

## Repository Structure

```
├── pixdit_core/      # Shared PixelDiT model definitions (c2i & t2i)
├── tools/            # Shared utilities (checkpoint download, GFLOPs computation)
├── c2i/              # Class-to-image
└── t2i/              # Text-to-image
```

## Compute GFLOPs

Measure single-forward-pass GFLOPs for any PixelDiT model (**run from project root**):

```bash
# C2I (ImageNet 256x256, default resolution)
python tools/compute_flops.py --config c2i/configs/pix256_xl.yaml
```

```bash
# T2I at 1024x1024
python tools/compute_flops.py --config t2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml --height 1024 --width 1024
```

## DualClock Phase 0

The experimental baseline reproduction, component profiler, regression
manifests, and semantic bypass parity harness live under
[`experiments/dualclock/`](experiments/dualclock/README.md). They do not modify
the released `pixdit_core/` path.

## Acknowledgements
We would like to thank the authors of [PixNerd](https://github.com/MCG-NJU/PixNerd) and [SANA](https://github.com/NVlabs/SANA) for sharing their code. We also thank the [SANA team](https://arxiv.org/pdf/2410.10629) for sharing their text-to-image training data.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yu2026pixeldit,
      title={PixelDiT: Pixel Diffusion Transformers for Image Generation},
      author={Yongsheng Yu and Wei Xiong and Weili Nie and Yichen Sheng and Shiqiu Liu and Jiebo Luo},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2026},
}
```
