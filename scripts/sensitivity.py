"""Sensitivity analysis (Tables 6 & 7): rule-hit threshold tau_r and planning
temperature tau, evaluated on the trained Full checkpoint (inference-time sweep)."""
import os, sys, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rgai.models import WorldModel
from rgai.rules import RuleLibrary
from rgai.metrics import evaluate
from scripts.configs import CONFIGS, ROOT
from scripts.run import load_datasets
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    cfg = CONFIGS[args.dataset](0)
    tr, va = load_datasets(cfg)
    run = os.path.join(ROOT, "runs", f"{cfg.name}_s0")
    ck = torch.load(os.path.join(run, "ckpt.pt"), map_location=cfg.train.device)
    m = WorldModel(cfg.model).to(cfg.train.device); m.load_state_dict(ck["model"], strict=False); m.eval()
    loader = DataLoader(va, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=tr.collate)

    print(f"\n=== Sensitivity: {cfg.name} ===")
    print("tau_r sweep (Acc@3):")
    for tau_r in [0.6, 0.7, 0.8, 0.9]:
        rules = RuleLibrary(cfg.rule, cfg.model.state_dim, cfg.model.n_actions, cfg.model.n_mental, cfg.train.device)
        rules.load_state_dict(ck["rules"]); rules.cfg.tau_r = tau_r
        r = evaluate(m, rules if len(rules) else None, loader, cfg.train, cfg.train.device, True, True)
        print(f"  tau_r={tau_r}: acc@3={r['acc@3']:.2f} hhar={r['hhar']:.2f} rhr={r['rule_hit_rate']:.1f}")

    print("tau (planning temperature) sweep (Acc@3):")
    for tau in [0.5, 1.0, 2.0]:
        rules = RuleLibrary(cfg.rule, cfg.model.state_dim, cfg.model.n_actions, cfg.model.n_mental, cfg.train.device)
        rules.load_state_dict(ck["rules"])
        c = cfg.train; old = c.plan_temp; c.plan_temp = tau
        r = evaluate(m, rules if len(rules) else None, loader, c, cfg.train.device, True, True)
        c.plan_temp = old
        print(f"  tau={tau}: acc@3={r['acc@3']:.2f} hhar={r['hhar']:.2f}")


if __name__ == "__main__":
    main()
