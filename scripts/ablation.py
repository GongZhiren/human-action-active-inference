"""Ablation study (Table 4).

Variants:
  Full            rules + latent intention m + generative consistency (VFE)
  w/o Rules       disable rule triggers at inference
  Rules w/o z     remove the discrete mental state (n_mental = 1), retrain
  -VFE            drop generative consistency (recon + transition KL), retrain
  Greedy          greedy rule selection (rule hit -> rule vote, no planning)

Inference-time variants reuse the trained Full checkpoint; the two training-time
variants retrain briefly.
"""
import os, sys, json, argparse, copy
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rgai.data import ArrayDataset, FrameDataset
from rgai.models import WorldModel
from rgai.rules import RuleLibrary
from rgai.metrics import evaluate
from rgai.trainer import Trainer
from scripts.configs import CONFIGS, ROOT
from scripts.run import load_datasets, set_seed
from torch.utils.data import DataLoader


def full_ckpt_eval(cfg, tr, va):
    run = os.path.join(ROOT, "runs", f"{cfg.name}_s0")
    ck = torch.load(os.path.join(run, "ckpt.pt"), map_location=cfg.train.device)
    m = WorldModel(cfg.model).to(cfg.train.device); m.load_state_dict(ck["model"], strict=False); m.eval()
    rules = RuleLibrary(cfg.rule, cfg.model.state_dim, cfg.model.n_actions, cfg.model.n_mental, cfg.train.device)
    rules.load_state_dict(ck["rules"])
    loader = DataLoader(va, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=tr.collate)
    out = {}
    out["Full"] = evaluate(m, rules if len(rules) else None, loader, cfg.train, cfg.train.device, True, True)
    out["w/o Rules"] = evaluate(m, None, loader, cfg.train, cfg.train.device, False, True)
    out["Greedy"] = evaluate(m, rules if len(rules) else None, loader, cfg.train, cfg.train.device, True, True, greedy_rule=True)
    return out


def retrain_variant(cfg, tr, va, variant):
    c = copy.deepcopy(cfg)
    if variant == "Rules w/o z":
        c.model.n_mental = 1
    elif variant == "-VFE":
        c.train.recon_w = 0.0; c.train.beta_kl = 0.0
    c.train.full_epochs = min(c.train.full_epochs, 10)
    set_seed(0)
    t = Trainer(c, tr, va, log=lambda *a: None)
    t.fit()
    loader = DataLoader(va, batch_size=c.train.batch_size, shuffle=False, collate_fn=tr.collate)
    return evaluate(t.model, t.rules if len(t.rules) else None, loader, c.train, c.train.device, True, True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    cfg = CONFIGS[args.dataset](0)
    tr, va = load_datasets(cfg)
    res = full_ckpt_eval(cfg, tr, va)
    for v in ["Rules w/o z", "-VFE"]:
        res[v] = retrain_variant(cfg, tr, va, v)
    print(f"\n=== Ablation: {cfg.name} ===")
    print(f"{'Variant':<14} {'Acc@1':>7} {'Acc@3':>7} {'Acc@5':>7} {'HHAR':>7} {'Lat(ms)':>8}")
    for k in ["Full", "w/o Rules", "Rules w/o z", "-VFE", "Greedy"]:
        r = res[k]
        print(f"{k:<14} {r['acc@1']:7.2f} {r['acc@3']:7.2f} {r['acc@5']:7.2f} {r['hhar']:7.2f} {r['latency_ms']:8.2f}")
    os.makedirs(os.path.join(ROOT, "runs", "ablation"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "runs", "ablation", f"{cfg.name}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
