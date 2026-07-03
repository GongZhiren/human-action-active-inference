"""Two-stage wake-sleep trainer (Alg. 1 & 3).

Stage 1: blockwise pretraining -- minimize VFE only for a fast warm-up.
Stage 2: full wake-sleep -- wake updates on real data under the joint objective
         (Eq. 6), then rule growing/refinement, then a light sleep replay pass.
"""
from __future__ import annotations
import time, math, os
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .models import WorldModel
from .rules import RuleLibrary
from .config import Config
from .engine import compute_losses, harvest_stats
from .metrics import evaluate


def _warmup_lambda(step, warmup):
    return min(1.0, (step + 1) / max(warmup, 1))


def _lr_factor(step, warmup, total, min_ratio=0.1):
    """Linear warmup then cosine decay to min_ratio."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    prog = min(max(prog, 0.0), 1.0)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))


class Trainer:
    def __init__(self, cfg: Config, train_ds, val_ds, log=print):
        self.cfg = cfg
        self.device = cfg.train.device
        self.log = log
        self.model = WorldModel(cfg.model).to(self.device)
        self.rules = RuleLibrary(cfg.rule, cfg.model.state_dim, cfg.model.n_actions,
                                 cfg.model.n_mental, self.device)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                                     num_workers=cfg.train.num_workers, collate_fn=train_ds.collate,
                                     pin_memory=True)
        # smaller loader for fast per-epoch monitoring (full val used for final eval)
        if len(val_ds) > 4000:
            mon = Subset(val_ds, list(range(0, len(val_ds), len(val_ds) // 4000)))
        else:
            mon = val_ds
        self.mon_loader = DataLoader(mon, batch_size=cfg.train.batch_size, shuffle=False,
                                     num_workers=cfg.train.num_workers, collate_fn=train_ds.collate,
                                     pin_memory=True)
        n = sum(p.numel() for p in self.model.parameters())
        self.log(f"[model] backbone={cfg.model.backbone} params={n/1e6:.2f}M")
        # inverse-frequency class weights for the action CE (boosts rare-action HHAR)
        pw = cfg.train.class_balance_pow
        self.sample_w = None
        if pw > 0:
            acts = np.asarray(train_ds.act)
            flat = acts.reshape(-1)
            freq = np.bincount(flat, minlength=cfg.model.n_actions).astype(np.float64) + 1.0
            w = (freq.sum() / freq) ** pw
            w = np.clip(w / w.mean(), 0.3, 6.0)                # keep dominant class learnable
            self.model.class_weight.copy_(torch.tensor(w, dtype=torch.float32, device=self.device))
            self.log(f"[class-balance] pow={pw} weight range [{w.min():.2f},{w.max():.2f}]")
            # mildly oversample windows whose predicted steps contain rare actions
            # (capped, so the dominant classes are not starved).
            fut = acts[:, cfg.train.hist_len:]                 # [N,P] target actions
            sw = w[fut].max(axis=1)
            sw = np.clip(sw / np.median(sw), 1.0, 3.0)
            self.sample_w = torch.tensor(sw, dtype=torch.double)

    def _loader(self, ds, shuffle=True, bs=None):
        sampler = None
        if shuffle and self.sample_w is not None and ds is self.train_ds:
            sampler = WeightedRandomSampler(self.sample_w, len(self.sample_w), replacement=True)
            shuffle = False
        return DataLoader(ds, batch_size=bs or self.cfg.train.batch_size, shuffle=shuffle,
                          sampler=sampler, num_workers=self.cfg.train.num_workers,
                          collate_fn=self.train_ds.collate, pin_memory=True, drop_last=True)

    def _to_dev(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    # ------------------------------------------------------------------ #
    def pretrain(self):
        c = self.cfg.train
        opt = torch.optim.AdamW(self.model.parameters(), lr=c.pretrain_lr, weight_decay=c.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=c.amp)
        N = len(self.train_ds)
        block = max(1, N // c.pretrain_blocks)
        step = 0
        for b in range(c.pretrain_blocks):
            idx = list(range(b * block, min((b + 1) * block, N)))
            loader = self._loader(Subset(self.train_ds, idx))
            for ep in range(c.pretrain_epochs):
                for batch in loader:
                    batch = self._to_dev(batch)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=c.amp):
                        loss, m = compute_losses(self.model, None, batch, c, stage="pretrain")
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
                    for g in opt.param_groups:
                        g["lr"] = c.pretrain_lr * _warmup_lambda(step, 100)
                    scaler.step(opt); scaler.update()
                    step += 1
            self.log(f"[pretrain] block {b+1}/{c.pretrain_blocks} vfe={m['vfe']:.3f} "
                     f"recon={m['recon']:.3f} ce={m['post_ce']:.3f}")

    # ------------------------------------------------------------------ #
    def wake_epoch(self, opt, scaler, step0, warmup, total):
        c = self.cfg.train
        loader = self._loader(self.train_ds)
        step = step0
        agg = {}
        for batch in loader:
            batch = self._to_dev(batch)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=c.amp):
                loss, m = compute_losses(self.model, self.rules, batch, c, stage="full")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
            for g in opt.param_groups:
                g["lr"] = c.full_lr * _lr_factor(step, warmup, total)
            scaler.step(opt); scaler.update()
            step += 1
            for k, v in m.items():
                agg[k] = agg.get(k, 0) + v
        n = max(1, len(loader))
        return step, {k: v / n for k, v in agg.items()}

    @torch.no_grad()
    def grow_rules(self, max_batches=40, refine=False):
        c = self.cfg.train
        loader = self._loader(self.train_ds, shuffle=True)
        Ss, Ms, As, Fs = [], [], [], []
        for i, batch in enumerate(loader):
            batch = self._to_dev(batch)
            S, M, A, Fe = harvest_stats(self.model, batch, c)
            Ss.append(S); Ms.append(M); As.append(A); Fs.append(Fe)
            if i + 1 >= max_batches:
                break
        S = torch.cat(Ss); M = torch.cat(Ms); A = torch.cat(As); Fe = torch.cat(Fs)
        if refine and len(self.rules) > 0:
            self.rules.refine(S, M, A, Fe)
        else:
            self.rules.build_from_stats(S, M, A, Fe)

    def sleep_epoch(self, opt, scaler, n_steps=100):
        """Generative replay: reconstruct-consistency + rule refine on replayed data.
        We reuse real mini-batches as replay seeds and minimize the joint objective
        (imagination-consistent), then refine rules."""
        c = self.cfg.train
        loader = self._loader(self.train_ds)
        it = iter(loader)
        for i in range(n_steps):
            try:
                batch = next(it)
            except StopIteration:
                break
            batch = self._to_dev(batch)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=c.amp):
                loss, _ = compute_losses(self.model, self.rules, batch, c, stage="full")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
            scaler.step(opt); scaler.update()
        self.grow_rules(refine=True)

    # ------------------------------------------------------------------ #
    def fit(self):
        c = self.cfg.train
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        t_start = time.time()
        self.log("=== Stage 1: blockwise pretraining ===")
        self.pretrain()
        self.log("=== Stage 2: full wake-sleep ===")
        opt = torch.optim.AdamW(self.model.parameters(), lr=c.full_lr, weight_decay=c.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=c.amp)
        steps_per_epoch = max(1, len(self.train_ds) // c.batch_size)
        total = c.full_epochs * steps_per_epoch
        warmup = min(c.warmup_steps, total // 5)
        step = 0
        best = -1.0
        best_state = None
        hist = []
        for ep in range(c.full_epochs):
            self.model.train()
            step, m = self.wake_epoch(opt, scaler, step, warmup, total)
            # grow rules once model is warm, then refine each epoch
            if ep == max(1, c.full_epochs // 5):
                self.grow_rules(refine=False)
                self.log(f"[rules] grew rule bank: RC={len(self.rules)}")
            elif ep > max(1, c.full_epochs // 5):
                self.sleep_epoch(opt, scaler, n_steps=80)
            res = evaluate(self.model, self.rules if len(self.rules) else None,
                           self.mon_loader, c, self.device)
            hist.append({"epoch": ep, **{f"tr_{k}": v for k, v in m.items()}, **res, "RC": len(self.rules)})
            self.log(f"[ep {ep:02d}] loss={m.get('loss',0):.3f} vfe={m.get('vfe',0):.3f} "
                     f"efe={m.get('efe',0):.3f} | acc@1={res['acc@1']:.2f} acc@3={res['acc@3']:.2f} "
                     f"acc@5={res['acc@5']:.2f} hhar={res['hhar']:.2f} RC={len(self.rules)} "
                     f"lat={res['latency_ms']:.2f}ms")
            # model selection balances top-k accuracy and rare-action accuracy
            # (HHAR), so the saved model is strong on both frequent and critical
            # low-frequency actions.  A tiny epoch preference breaks ties toward
            # later epochs (mature, matched rule bank; rules grow at ep=E//5).
            score = (res["acc@1"] + res["acc@3"] + res["acc@5"]) / 3.0 + 0.3 * res["hhar"] + 5e-4 * ep
            if score > best:
                best = score
                best_state = {"model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                              "rules": self.rules.state_dict()}
        ct = (time.time() - t_start) / 3600.0
        pm = torch.cuda.max_memory_allocated() / 1e6 if self.device.startswith("cuda") else 0.0
        if best_state is not None:
            self.model.load_state_dict(best_state["model"])
            self.rules.load_state_dict(best_state["rules"])
        return dict(history=hist, convergence_time_h=ct, peak_mem_mb=pm, best_acc3=best)
