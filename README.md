<div align="center">

# Learning Human Habits with Rule-Guided Active Inference

### A library of latent-grounded symbolic rules, learned jointly with a generative world model through a biologically inspired wake–sleep loop

**RGAI turns deliberate active-inference planning into cheap, reusable habits:
familiar contexts fire a rule and act instantly, novel contexts fall back on full
expected-free-energy planning — so decisions get faster without ever overriding a
confident plan.**

<table>
  <tr>
    <td align="center">
      <a href="https://openreview.net/forum?id=FZXwkBH6s7"><strong>📄 Read the paper</strong></a><br>
      <sub>ICLR 2026 · method &amp; results</sub>
    </td>
    <td align="center">
      <a href="https://gongzhiren.github.io/ActiveInference-website/"><strong>🌐 Explore the project</strong></a><br>
      <sub>Visual story and highlights</sub>
    </td>
    <td align="center">
      <a href="#training"><strong>⚡ Quick start</strong></a><br>
      <sub>Train one dataset in a line</sub>
    </td>
    <td align="center">
      <a href="#datasets"><strong>🧩 Get the data</strong></a><br>
      <sub>Four domains, one pipeline</sub>
    </td>
  </tr>
</table>

[![Paper](https://img.shields.io/badge/OpenReview-FZXwkBH6s7-b31b1b.svg)](https://openreview.net/forum?id=FZXwkBH6s7)
[![Venue](https://img.shields.io/badge/ICLR-2026-8a2be2.svg)](https://openreview.net/forum?id=FZXwkBH6s7)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Overview](#overview) · [Highlights](#highlights) · [Install](#installation) · [Datasets](#datasets) · [Training](#training) · [Evaluation](#evaluation--analysis) · [Citation](#citation)

</div>

<p align="center">
  <img src="assets/overview.png" alt="Rule-Guided Active Inference: generative world model, habitual rule construction and activation, and the wake-sleep free-energy loop" width="100%">
</p>

<p align="center"><em>A split latent world model, a habitual rule library, and a wake–sleep loop that grows, reinforces, and prunes rules — fusing planning and habit under one free-energy objective.</em></p>

## News

- **2026 — ICLR release.** Official RGAI implementation: the full wake–sleep trainer,
  four data-domain pipelines, and the ablation / sensitivity / trade-off analysis scripts.

## Overview

Active inference (AIF) casts perception and action as the minimization of a single
quantity — **free energy** — but planning every action through expected free energy
(EFE) is expensive and ignores the fact that much of human behavior is **habitual**:
fast, automatic responses to familiar situations.

**RGAI** (Rule-Guided Active Inference) closes this gap. It maintains a small library
of symbolic rules, each anchored to a region of the agent's latent state, capturing
recurring stimulus→action habits. At decision time:

- **Familiar context** → a rule fires and triggers an action *instantly*, bypassing planning.
- **Novel context** → no rule matches, and the agent falls back on full EFE planning.

Rules are discovered, reinforced, and pruned online through a wake–sleep loop, so the
agent continually distills its own deliberate behavior into cheap habits.

```
Z_t = (S_t, m_t)          latent split: continuous world state S_t + discrete mental state m_t
p_φ(O_t | S_t)            world-model decoder (likelihood)
p_φ(S_t | S_{t-1}, a)     state-transition prior (residual: S_t = S_{t-1} + Δ)
p_φ(m_t | m_{t-1}, S_t)   sticky mental-state transition
π_φ(a_t | S_t, m_t)       rule-augmentable policy head
r : (S*_r, m*_r) ↦ a_r    a rule (Gaussian-kernel anchor + mode + action + confidence ρ_r)
```

The inference-time distribution fuses planning and habit as a **bounded, margin-gated
blend**: a confident rule sharpens the action only where the planner is genuinely
uncertain, so rules accelerate decisions without ever overriding a confident plan.

## Highlights

<table>
  <tr>
    <td align="center"><strong>4 domains</strong><br><sub>driving · sports · medical · vision</sub></td>
    <td align="center"><strong>Wake–sleep</strong><br><sub>grow · reinforce · prune rules online</sub></td>
    <td align="center"><strong>Rules skip planning</strong><br><sub>matched habits cut latency</sub></td>
    <td align="center"><strong>One objective</strong><br><sub>F = VFE + η·EFE + γ·KL</sub></td>
  </tr>
</table>

- 🧠 **Unified free-energy objective** — `F = VFE + η·EFE + γ·KL`, optimized end to end.
- ⚡ **Rules bypass planning** — matched habits skip the EFE rollout, cutting latency.
- ♻️ **Wake–sleep rule lifecycle** — grow on recurring low-free-energy patterns, reinforce, prune.
- 🎛️ **Split latent** `(S, m)` — continuous world state + discrete mental state (context/intent).
- 🧩 **Four domains, one framework** — driving, sports, medical dialogue, and vision-based control.

<details>
<summary><strong>Repository layout</strong></summary>

```
rgai/                   core library
├── config.py           dataclass configurations
├── models.py           encoders (Transformer / BiGRU / Evidence2Vec / CNN) + world-model heads
├── rules.py            symbolic rule library (EM-style Gaussian mixture over latents)
├── engine.py           VFE / EFE, wake–sleep losses, EFE planning, hybrid inference
├── trainer.py          two-stage wake–sleep trainer
├── metrics.py          Acc@1/3/5, HHAR, latency, peak memory
└── data.py             unified array / frame datasets
data_pipeline/          dataset preprocessors
├── prep_nba.py  prep_car.py  prep_ddx.py  prep_atari.py
scripts/
├── run.py              training entry point
├── run_matrix.sh       train all datasets × seeds
├── configs.py          per-dataset configurations
├── collect_results.py  aggregate the results table across seeds
├── ablation.py         ablation study
├── sensitivity.py      τ_r / planning-temperature sensitivity
├── rule_tradeoff.py    rule-bank size vs. accuracy / latency
└── training_dynamics.py  per-epoch free-energy & rule-bank curves
```

</details>

## Installation

```bash
git clone https://github.com/GongZhiren/human-action-active-inference.git
cd human-action-active-inference
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and a CUDA-capable GPU (training uses AMP).

## Datasets

| Domain | Source | Backbone | Actions | Mental states |
|---|---|---|---|---|
| **NBA SportVU** | [linouk23/NBA-Player-Movements](https://github.com/linouk23/NBA-Player-Movements) | Transformer | 7 | 3 |
| **Car-Following** | [RomainLITUD/Car-Following-Dataset-HV-vs-AV](https://github.com/RomainLITUD/Car-Following-Dataset-HV-vs-AV) | BiGRU | 7 | 2 |
| **DDXPlus** (URTI) | [mila-iqia/ddxplus](https://github.com/mila-iqia/ddxplus) | Evidence2Vec + Transformer | 225 | 5 |
| **Atari-HEAD** | [Atari-HEAD (Zenodo 3451402)](https://zenodo.org/records/3451402) | CNN enc–dec + Transformer | 18 | 4 |

Place each raw download under `data_raw/`, then build the processed tensors:

```bash
cd data_pipeline
python prep_car.py       # -> data_proc/car/
python prep_nba.py       # -> data_proc/nba/
python prep_ddx.py       # -> data_proc/ddxplus/
python prep_atari.py     # -> data_proc/atari_berzerk/   (use --raw for other games)
```

Each preprocessor exposes `--raw` / `--out` and documents its feature construction,
action definitions, and windowing at the top of the file.

## Training

```bash
# one dataset / one seed
python scripts/run.py --dataset car     --seed 0
python scripts/run.py --dataset nba     --seed 0
python scripts/run.py --dataset ddxplus --seed 0
python scripts/run.py --dataset atari   --seed 0 --game berzerk

# everything (4 datasets × 3 seeds, seeds concurrent per dataset)
bash scripts/run_matrix.sh

# aggregate the results table across seeds
python scripts/collect_results.py
```

Training follows a **two-stage schedule**: blockwise VFE **pretraining** for a fast
warm-up, then **full wake–sleep** under the joint objective `F = VFE + η·EFE + γ·KL`,
with the rule library grown and consolidated online. Checkpoints and per-epoch logs
are written to `runs/<dataset>_s<seed>/`.

## Evaluation & analysis

Reproduce the paper's tables and figures from trained checkpoints:

```bash
python scripts/ablation.py        --dataset car      # variant ablations
python scripts/sensitivity.py     --dataset nba      # τ_r / planning-temperature sweeps
python scripts/rule_tradeoff.py   --dataset atari    # rule-bank size vs. accuracy / latency
python scripts/training_dynamics.py                  # free-energy & rule-bank curves
```

**Metrics**

- **Acc@1/3/5** — mean per-step accuracy over the next 1 / 3 / 5 autoregressive steps.
- **HHAR** (High-Hit Action Ratio) — accuracy on *critical low-frequency* actions
  (below the mean frequency of observed actions), measuring how well rare but decisive
  maneuvers are captured.
- **Latency** — ms per decision step · **CT** — training hours · **PM** — peak memory · **RC** — rule count.

## Configuration

All hyperparameters live in `rgai/config.py` (defaults) and `scripts/configs.py`
(per-dataset overrides), mirroring the appendix tables — backbone sizes, the
free-energy weights `η, γ`, planning horizon / beam width, the rule-kernel bandwidth
and thresholds `τ_r, δ_F, δ_sup, δ_conf`, and the fusion cap `fusion_alpha`. Data and
checkpoint roots resolve to the repository directory by default; override with the
`RGAI_ROOT` environment variable.

## Citation

```bibtex
@inproceedings{zhiren2026learning,
  title     = {Learning Human Habits with Rule-Guided Active Inference},
  author    = {Gong, Zhiren and Yang, Chao and Ren, Wendi and Li, Shuang},
  booktitle = {The Fourteenth International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=FZXwkBH6s7}
}
```

## License

Released under the [MIT License](LICENSE).
