"""Per-dataset configurations (Tables 2 & 3, Appendix C.6/C.7)."""
import os
from rgai.config import Config, ModelConfig, RuleConfig, TrainConfig

# Repository root: data lives in <root>/data_proc, checkpoints in <root>/runs.
# Override with the RGAI_ROOT environment variable if desired.
ROOT = os.environ.get("RGAI_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def car_config(seed=0):
    c = Config(name="car")
    c.data_dir = f"{ROOT}/data_proc/car"
    c.model = ModelConfig(backbone="bigru", obs_dim=3, state_dim=64, hidden_dim=128,
                          n_layers=3, n_heads=4, n_actions=7, n_mental=2)
    c.rule = RuleConfig(sigma=1.0, tau_r=0.7, delta_F=0.25, delta_sup=0.75, delta_conf=0.75,
                        conf_init=0.3, max_rules=8, fusion_alpha=0.4)
    c.train = TrainConfig(pretrain_blocks=5, pretrain_epochs=1, pretrain_lr=1e-4,
                          full_epochs=15, full_lr=1e-4, weight_decay=0.01, warmup_steps=4000,
                          batch_size=64, eta=0.01, gamma=1.0, horizon=4, beam_width=6,
                          plan_temp=0.5, hist_len=10, pred_len=5, seed=seed)
    return c


def nba_config(seed=0):
    c = Config(name="nba")
    c.data_dir = f"{ROOT}/data_proc/nba"
    c.model = ModelConfig(backbone="transformer", obs_dim=22, state_dim=96, hidden_dim=256,
                          n_layers=4, n_heads=4, n_actions=7, n_mental=3)
    c.rule = RuleConfig(sigma=1.0, tau_r=0.8, delta_F=0.25, delta_sup=0.75, delta_conf=0.75,
                        conf_init=0.3, max_rules=8, fusion_alpha=0.05)
    c.train = TrainConfig(pretrain_blocks=5, pretrain_epochs=1, pretrain_lr=1e-3,
                          full_epochs=15, full_lr=3e-4, weight_decay=0.01, warmup_steps=2000,
                          batch_size=64, eta=0.05, gamma=1.0, horizon=4, beam_width=6,
                          plan_temp=1.0, plan_look_w=0.0, hist_len=10, pred_len=5, seed=seed, chg_weight=2.0)
    return c


def ddx_config(seed=0):
    c = Config(name="ddxplus")
    c.data_dir = f"{ROOT}/data_proc/ddxplus"
    c.model = ModelConfig(backbone="evidence2vec", obs_dim=0, state_dim=128, hidden_dim=256,
                          n_layers=3, n_heads=4, n_actions=225, n_mental=5,
                          vocab_size=0, extra_dim=0)  # vocab/extra set from data meta
    c.rule = RuleConfig(sigma=1.5, tau_r=0.8, delta_F=0.5, delta_sup=0.75, delta_conf=0.75,
                        conf_init=0.3, max_rules=256)
    c.train = TrainConfig(pretrain_blocks=5, pretrain_epochs=1, pretrain_lr=1e-3,
                          full_epochs=15, full_lr=3e-4, weight_decay=0.01, warmup_steps=16000,
                          batch_size=64, eta=0.01, gamma=10.0, horizon=4, beam_width=6,
                          plan_temp=1.0, hist_len=10, pred_len=5, seed=seed, chg_weight=0.0, mask_repeat_below=223)
    return c


def atari_config(seed=0, game="berzerk"):
    c = Config(name=f"atari_{game}")
    c.data_dir = f"{ROOT}/data_proc/atari_{game}"
    c.model = ModelConfig(backbone="cnn", obs_dim=0, state_dim=128, hidden_dim=256,
                          n_layers=2, n_heads=4, n_actions=18, n_mental=4,
                          img_size=128, img_channels=1)
    c.rule = RuleConfig(sigma=2.0, tau_r=0.7, delta_F=5.0, delta_sup=0.75, delta_conf=0.75,
                        conf_init=0.3, max_rules=120, fusion_alpha=0.35)
    c.train = TrainConfig(pretrain_blocks=5, pretrain_epochs=1, pretrain_lr=1e-4,
                          full_epochs=15, full_lr=1e-4, weight_decay=0.01, warmup_steps=2000,
                          batch_size=32, eta=0.2, gamma=5.0, horizon=4, beam_width=6,
                          plan_temp=1.0, hist_len=10, pred_len=5, seed=seed, num_workers=8, chg_weight=2.0, class_balance_pow=0.9)
    return c


CONFIGS = {"car": car_config, "nba": nba_config, "ddxplus": ddx_config, "atari": atari_config}
