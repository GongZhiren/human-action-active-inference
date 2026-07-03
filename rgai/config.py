"""Configuration dataclasses for RGAI.

All dataset-specific settings follow Appendix C.7 (Tables 2 & 3) of the paper.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple
import json


@dataclass
class ModelConfig:
    backbone: str = "transformer"      # transformer | bigru | evidence2vec | cnn
    obs_dim: int = 16                  # dimension of per-step observation features O_t
    state_dim: int = 64                # continuous latent S_t dimension
    hidden_dim: int = 128
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    n_actions: int = 7                 # size of discrete action space
    n_mental: int = 3                  # K, number of discrete mental states m
    # cnn / vision
    img_size: int = 128
    img_channels: int = 1
    # evidence2vec (ddxplus)
    vocab_size: int = 0                # number of distinct evidence tokens (0 = not used)
    topk_diag: int = 50                # top-K posterior over candidate diagnoses
    extra_dim: int = 0                 # numeric side-features per step (posterior + entropy)


@dataclass
class RuleConfig:
    sigma: float = 1.0                 # Gaussian kernel bandwidth for kappa
    tau_r: float = 0.8                 # rule-hit threshold on kappa (Table 6)
    delta_F: float = 0.25              # free-energy improvement threshold to grow a rule
    delta_sup: float = 0.75            # support (recurrence) threshold
    delta_conf: float = 0.75           # confidence prune threshold
    conf_init: float = 0.3             # rho_r initial confidence
    conf_step: float = 0.1             # delta_conf increment on reinforcement
    max_rules: int = 256               # cap on rule-bank size (RC)
    merge_dist: float = 0.5            # merge new rule into nearest if closer than this (in kappa)
    fusion_beta: float = 3.0          # weight of rule log-vote added to the plan distribution
    fusion_alpha: float = 0.2         # max blend weight toward the rule prior (bounded, non-degrading)


@dataclass
class TrainConfig:
    # two-stage schedule (Alg 3)
    pretrain_blocks: int = 5
    pretrain_epochs: int = 1
    pretrain_lr: float = 1e-3
    full_epochs: int = 15
    full_lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    batch_size: int = 64
    grad_clip: float = 1.0
    # free-energy weights (Eq 6)
    eta: float = 0.05                  # EFE weight
    gamma: float = 1.0                 # mental-state KL weight
    beta_kl: float = 1.0               # state-transition KL weight inside VFE
    recon_w: float = 1.0               # observation reconstruction weight
    act_w: float = 1.0                 # action cross-entropy weight
    chg_weight: float = 3.0            # extra CE weight on change-point steps (a_t != a_{t-1})
    class_balance_pow: float = 0.0     # inverse-frequency class weighting exponent (0 = off); boosts rare-action (HHAR) accuracy
    # planning (Eq 3, Eq 5)
    horizon: int = 4                   # EFE planning horizon H
    beam_width: int = 6                # beam width K
    plan_temp: float = 1.0             # planning temperature tau (Table 7)
    plan_look_w: float = 0.1           # weight of the EFE look-ahead term (0 = policy-only fallback)
    # task
    hist_len: int = 10                 # history window L
    pred_len: int = 5                  # prediction window P
    mask_repeat_below: int = 0         # during rollout, mask already-taken actions with id < this (0=off); DDXPlus: no re-asking an evidence
    # misc
    seed: int = 0
    device: str = "cuda"
    amp: bool = True
    num_workers: int = 3


@dataclass
class Config:
    name: str = "nba"
    model: ModelConfig = field(default_factory=ModelConfig)
    rule: RuleConfig = field(default_factory=RuleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data_dir: str = ""
    out_dir: str = ""

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        c = Config()
        c.name = d.get("name", c.name)
        c.data_dir = d.get("data_dir", "")
        c.out_dir = d.get("out_dir", "")
        if "model" in d:
            c.model = ModelConfig(**d["model"])
        if "rule" in d:
            c.rule = RuleConfig(**d["rule"])
        if "train" in d:
            c.train = TrainConfig(**d["train"])
        return c
