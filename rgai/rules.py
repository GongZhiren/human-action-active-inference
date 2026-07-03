"""Symbolic rule library (Sec. 4.1, Appendix B).

A rule is an anchored condition-action pair  f: (S*_r, m*_r) => a_r  with a
confidence weight rho_r in [0,1].  The library is a lightweight, EM-style
Gaussian-mixture approximation over latent contexts (S_t, m_t):

    kappa(S^MAP, S*_r) = exp( -||S^MAP - S*_r||^2 / (2 sigma^2) )   (Gaussian kernel)

A rule is *active* when kappa >= tau_r and m^MAP == m*_r.  Rule anchors are
updated as VFE-weighted centroids of their assigned latents (M-step).
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

from .config import RuleConfig


class RuleLibrary:
    def __init__(self, cfg: RuleConfig, state_dim: int, n_actions: int, n_mental: int, device="cuda"):
        self.cfg = cfg
        self.ds = state_dim
        self.n_actions = n_actions
        self.n_mental = n_mental
        self.device = device
        self.anchors = torch.zeros(0, state_dim, device=device)   # S*_r
        self.m_star = torch.zeros(0, dtype=torch.long, device=device)
        self.a_star = torch.zeros(0, dtype=torch.long, device=device)
        self.rho = torch.zeros(0, device=device)                  # confidence
        self.support = torch.zeros(0, device=device)              # accumulated responsibility
        self.sigma = cfg.sigma                                     # kernel bandwidth (calibrated to data)

    def __len__(self):
        return self.anchors.shape[0]

    # ------------------------------------------------------------------ #
    def kappa(self, S):
        """Gaussian kernel between S [B,ds] and all anchors -> [B,R]."""
        if len(self) == 0:
            return torch.zeros(S.shape[0], 0, device=S.device)
        d2 = torch.cdist(S, self.anchors) ** 2
        return torch.exp(-d2 / (2 * self.sigma ** 2))

    def activate(self, S_map, m_map):
        """Return (vote [B,A], hit [B] bool, kappa [B,R]).

        vote(a) = sum_{r: a_r=a} kappa_r * 1{m match} * rho_r   (weighted rule vote,
        Eq. after 4.1, unnormalized).  A rule is active (hit) when kappa >= tau_r
        and m matches.
        """
        B = S_map.shape[0]
        A = self.n_actions
        vote = torch.zeros(B, A, device=S_map.device)
        if len(self) == 0:
            return vote, torch.zeros(B, dtype=torch.bool, device=S_map.device), \
                torch.zeros(B, 0, device=S_map.device)
        k = self.kappa(S_map)                                   # [B,R]
        m_match = (m_map.unsqueeze(1) == self.m_star.unsqueeze(0)).float()  # [B,R]
        w = k * m_match * self.rho.unsqueeze(0)                 # weighted responsibility
        idx = self.a_star.unsqueeze(0).expand(B, -1)
        vote.scatter_add_(1, idx, w)
        active = (k >= self.cfg.tau_r) & (m_match > 0)
        hit = active.any(dim=1)
        return vote, hit, k

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def build_from_stats(self, S, m, a, fe):
        """Grow the rule bank from wake-phase statistics (Alg. 1, lines 8-10).

        Group low-free-energy (reliably explained) triplets (S^MAP, m^MAP, a)
        and form prototypes by leader clustering; anchors are VFE-weighted
        centroids, confidences scale with support.  Caps at max_rules.
        """
        cfg = self.cfg
        S = S.to(self.device); m = m.to(self.device); a = a.to(self.device); fe = fe.to(self.device)
        # keep triplets whose free energy is in the low (reliable) tail.
        # delta_F is a quantile when in (0,1]; datasets that specify it as an
        # absolute free-energy margin (>1, e.g. Atari) fall back to the median.
        q = cfg.delta_F if 0.0 < cfg.delta_F <= 1.0 else 0.5
        thr = torch.quantile(fe, q) if fe.numel() > 0 else 0.0
        keep = fe <= thr
        S, m, a, fe = S[keep], m[keep], a[keep], fe[keep]
        w_all = torch.exp(-(fe - fe.min())) if fe.numel() else fe

        # provisional bandwidth for leader-clustering merges (median pairwise)
        if S.shape[0] > 4:
            samp = S[torch.randperm(S.shape[0], device=self.device)[:2048]]
            d = torch.cdist(samp, samp)
            self.sigma = float(torch.median(d[d > 0]).clamp_min(1e-3)) * 0.5
        anchors, m_star, a_star, rho, support = [], [], [], [], []
        merge_d2 = -2 * self.sigma ** 2 * torch.log(torch.tensor(max(cfg.merge_dist, 1e-4)))
        for mm in range(self.n_mental):
            for aa in range(self.n_actions):
                sel = (m == mm) & (a == aa)
                n = int(sel.sum())
                if n < 2:
                    continue
                pts = S[sel]; wts = w_all[sel]
                # greedy leader clustering by squared distance
                order = torch.argsort(wts, descending=True)
                pts, wts = pts[order], wts[order]
                assigned = torch.zeros(pts.shape[0], dtype=torch.bool, device=self.device)
                for i in range(pts.shape[0]):
                    if assigned[i]:
                        continue
                    d2 = ((pts - pts[i]) ** 2).sum(1)
                    grp = (d2 <= merge_d2) & (~assigned)
                    assigned |= grp
                    gw = wts[grp]
                    centroid = (pts[grp] * gw.unsqueeze(1)).sum(0) / gw.sum().clamp_min(1e-8)
                    sup = float(grp.sum())
                    anchors.append(centroid)
                    m_star.append(mm); a_star.append(aa)
                    support.append(sup)
        if not anchors:
            return
        anchors = torch.stack(anchors)
        support = torch.tensor(support, device=self.device)
        m_star = torch.tensor(m_star, device=self.device, dtype=torch.long)
        a_star = torch.tensor(a_star, device=self.device, dtype=torch.long)
        # confidence: normalized support, floored at conf_init
        rho = torch.clamp(support / support.max().clamp_min(1), min=cfg.conf_init)
        # support threshold + cap
        sup_thr = torch.quantile(support.float(), 1 - min(1.0, cfg.max_rules / max(len(support), 1))) \
            if len(support) > cfg.max_rules else torch.tensor(0.0, device=self.device)
        keepr = support >= sup_thr
        anchors, m_star, a_star, rho, support = \
            anchors[keepr], m_star[keepr], a_star[keepr], rho[keepr], support[keepr]
        if len(anchors) > cfg.max_rules:
            top = torch.argsort(rho, descending=True)[: cfg.max_rules]
            anchors, m_star, a_star, rho, support = \
                anchors[top], m_star[top], a_star[top], rho[top], support[top]
        self.anchors, self.m_star, self.a_star, self.rho, self.support = \
            anchors, m_star, a_star, rho, support
        # calibrate activation bandwidth so ~half the states fire: set sigma so a
        # state at the median distance-to-nearest-anchor has kappa = tau_r.
        if len(self) > 0:
            dn = torch.cdist(S, self.anchors).min(dim=1).values
            med = torch.median(dn).clamp_min(1e-3)
            self.sigma = float(med / (-2 * torch.log(torch.tensor(cfg.tau_r))).sqrt())

    @torch.no_grad()
    def refine(self, S, m, a, fe):
        """Sleep-phase refinement (Alg. 1, line 19): EM M-step on anchors and a
        confidence = *reliability* update -- rho_r is the precision of rule r
        (fraction of its activations whose observed action matches a_r).  Rules
        that never activate or are unreliable are pruned; reliable ones persist.
        """
        if len(self) == 0:
            return
        cfg = self.cfg
        S = S.to(self.device); m = m.to(self.device); a = a.to(self.device)
        k = self.kappa(S)                                   # [B,R]
        active = (k >= cfg.tau_r) & (m.unsqueeze(1) == self.m_star.unsqueeze(0))  # [B,R]
        a_match = (a.unsqueeze(1) == self.a_star.unsqueeze(0))
        n_active = active.float().sum(0)                    # [R] how often rule fires
        n_correct = (active & a_match).float().sum(0)       # [R] fires and correct
        precision = n_correct / n_active.clamp_min(1e-6)    # reliability
        # EM M-step: move anchor toward the mean of correctly-explained latents
        resp = (active & a_match).float()
        denom = resp.sum(0)
        new_anchor = (resp.t() @ S) / denom.clamp_min(1e-6).unsqueeze(1)
        upd = denom > 0
        self.anchors[upd] = 0.8 * self.anchors[upd] + 0.2 * new_anchor[upd]
        # recalibrate bandwidth to the (possibly drifted) current latent scale
        dn = torch.cdist(S, self.anchors).min(dim=1).values
        med = torch.median(dn).clamp_min(1e-3)
        self.sigma = 0.5 * self.sigma + 0.5 * float(med / (-2 * torch.log(torch.tensor(cfg.tau_r))).sqrt())
        # confidence <- reliability (EMA); rules that fire keep it grounded
        fired = n_active > 0
        self.rho = torch.where(fired, 0.5 * self.rho + 0.5 * precision, self.rho)
        # prune only clearly-unreliable rules that actually fire; always retain
        # the most reliable few so the bank never collapses to empty.
        keep = (~fired) | (self.rho >= 0.2)
        if keep.sum() < min(4, len(self)):
            top = torch.argsort(self.rho, descending=True)[: min(4, len(self))]
            keep = torch.zeros(len(self), dtype=torch.bool, device=self.device)
            keep[top] = True
        if 0 < keep.sum() < len(self):
            self.anchors, self.m_star, self.a_star, self.rho, self.support = \
                self.anchors[keep], self.m_star[keep], self.a_star[keep], \
                self.rho[keep], self.support[keep]

    def state_dict(self):
        return dict(anchors=self.anchors.cpu(), m_star=self.m_star.cpu(),
                    a_star=self.a_star.cpu(), rho=self.rho.cpu(), support=self.support.cpu(),
                    sigma=self.sigma)

    def load_state_dict(self, sd):
        self.anchors = sd["anchors"].to(self.device)
        self.m_star = sd["m_star"].to(self.device)
        self.a_star = sd["a_star"].to(self.device)
        self.rho = sd["rho"].to(self.device)
        self.support = sd["support"].to(self.device)
        self.sigma = float(sd.get("sigma", self.cfg.sigma))
