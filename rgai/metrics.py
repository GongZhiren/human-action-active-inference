"""Evaluation metrics: Acc@1/3/5, High-Hit Action Ratio (HHAR), latency, memory."""
from __future__ import annotations
import time
from typing import Dict
import numpy as np
import torch

from .engine import predict, _hist_batch


def acc_at_k(preds: np.ndarray, targets: np.ndarray, ks=(1, 3, 5)) -> Dict[str, float]:
    """preds/targets: [N,P]. Acc@k = fraction correct over the next k steps."""
    correct = (preds == targets)
    out = {}
    for k in ks:
        out[f"acc@{k}"] = float(correct[:, :k].mean()) * 100.0
    return out


def hhar(preds: np.ndarray, targets: np.ndarray, n_actions: int, ratio: float = 1.0,
         min_support: float = 0.005) -> float:
    """High-Hit Action Ratio: accuracy on low-frequency *critical* actions.

    Rare/critical = *observed* actions whose frequency is below the mean
    frequency of the observed actions and that have at least `min_support` of the
    data.  Defining "low-frequency" relative to the actions that actually occur
    (not the padded vocabulary) keeps HHAR meaningful when the action space is
    largely unused (e.g. DDXPlus's 225-way vocab with ~20 active actions).
    HHAR = accuracy on all (sample,step) whose ground-truth action is critical.
    """
    flat_t = targets.reshape(-1)
    flat_p = preds.reshape(-1)
    N = flat_t.shape[0]
    freq = np.bincount(flat_t, minlength=n_actions).astype(float)
    active = freq[freq > 0]
    thr = ratio * active.mean() if active.size else 0.0     # below-average among observed actions
    lo = min_support * N
    rare = set(np.where((freq >= lo) & (freq < thr))[0].tolist())
    if not rare:
        rare = set(np.where((freq > 0) & (freq < thr))[0].tolist())
    if not rare:
        return 0.0
    mask = np.isin(flat_t, list(rare))
    if mask.sum() == 0:
        return 0.0
    return float((flat_p[mask] == flat_t[mask]).mean()) * 100.0


@torch.no_grad()
def evaluate(model, rules, loader, cfg, device, use_rules=True, use_plan=True,
             greedy_rule=False, n_actions=None) -> Dict[str, float]:
    model.eval()
    all_p, all_t, all_hit = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = predict(model, rules, batch, cfg, use_rules, use_plan, greedy_rule)
        P = cfg.pred_len
        tgt = batch["act"][:, cfg.hist_len: cfg.hist_len + P]
        all_p.append(out["preds"].cpu().numpy())
        all_t.append(tgt.cpu().numpy())
        all_hit.append(out["hits"].cpu().numpy())
    preds = np.concatenate(all_p); targets = np.concatenate(all_t)
    hits = np.concatenate(all_hit)
    n_actions = n_actions or model.cfg.n_actions
    res = acc_at_k(preds, targets)
    # HHAR = accuracy on critical low-frequency actions over the Acc@3 horizon
    # (below-uniform frequency with >=0.5% support).  The Acc@3 window is the
    # standard mid-horizon reporting granularity; it decouples rare-action
    # recognition from the far-horizon rollout drift that dominates Acc@5.
    hh = min(3, preds.shape[1])
    res["hhar"] = hhar(preds[:, :hh], targets[:, :hh], n_actions)
    res["rule_hit_rate"] = float(hits.mean()) * 100.0
    res["latency_ms"] = measure_latency(model, rules, loader, cfg, device,
                                        use_rules, use_plan, greedy_rule)
    return res


@torch.no_grad()
def measure_latency(model, rules, loader, cfg, device, use_rules=True, use_plan=True,
                    greedy_rule=False, n_samples=64) -> float:
    """Average wall-clock latency per decision step (batch size 1)."""
    model.eval()
    times = []
    count = 0
    for batch in loader:
        batch = {k: v[:1].to(device) for k, v in batch.items()}
        # warmup once
        if count == 0:
            predict(model, rules, batch, cfg, use_rules, use_plan, greedy_rule)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
        t0 = time.perf_counter()
        predict(model, rules, batch, cfg, use_rules, use_plan, greedy_rule)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0 / cfg.pred_len)
        count += 1
        if count >= n_samples:
            break
    return float(np.mean(times)) if times else 0.0
