"""Free-energy engine: VFE/EFE computation, wake-sleep losses, EFE planning,
and the hybrid rule+planning inference used for multi-step action forecasting.

Objective (Eq. 6):  F_t = VFE_t + eta * EFE_t + gamma * KL(q(m_t)||q(m_{t-1}))
    VFE  =  reconstruction (-log p(O|S))  +  state-transition KL  +  action CE
    EFE  =  imitation on prior rollouts (matches test-time autoregressive use)
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import WorldModel
from .config import TrainConfig
from .rules import RuleLibrary


def gaussian_kl(mq, lvq, mp, lvp):
    """KL( N(mq,exp(lvq)) || N(mp,exp(lvp)) ) summed over last dim."""
    return 0.5 * (lvp - lvq + (lvq.exp() + (mq - mp) ** 2) / lvp.exp() - 1).sum(-1)


def std_normal_kl(mq, lvq):
    return 0.5 * (mq ** 2 + lvq.exp() - 1 - lvq).sum(-1)


def cat_kl(logq, logp):
    q = logq.softmax(-1)
    return (q * (logq.log_softmax(-1) - logp.log_softmax(-1))).sum(-1)


def _hist_batch(batch, L):
    """Slice the encoder input to the history window (steps 0..L-1)."""
    out = {}
    if "obs" in batch:
        out["obs"] = batch["obs"][:, :L]
    if "frames" in batch:
        out["frames"] = batch["frames"][:, :L]
    if "tok" in batch:
        out["tok"] = batch["tok"][:, :L]
        out["extra"] = batch["extra"][:, :L]
    out["act_prev"] = batch["act_prev"][:, :L]
    return out


def recon_target(model: WorldModel, batch, t):
    """Observation reconstruction target at absolute step t."""
    bb = model.cfg.backbone
    if bb == "cnn":
        return batch["frames"][:, t]
    if bb == "evidence2vec":
        return batch["extra"][:, t]
    return batch["obs"][:, t]


def recon_loss(model, pred, target):
    if model.cfg.backbone == "cnn":
        # pred are logits (AMP-safe); scale down to keep the pixel loss balanced
        return F.binary_cross_entropy_with_logits(pred, target, reduction="none").flatten(1).mean(-1)
    return ((pred - target) ** 2).sum(-1)


def compute_losses(model: WorldModel, rules: Optional[RuleLibrary], batch,
                   cfg: TrainConfig, stage: str = "full") -> Tuple[torch.Tensor, Dict]:
    """Return (scalar loss, metrics).  stage='pretrain' uses VFE only (Eq. 2)."""
    L, P = cfg.hist_len, cfg.pred_len
    device = batch["act"].device
    B = batch["act"].shape[0]

    # ---- encode history window -> per-step posterior q(S_t,m_t|H_t) --------- #
    S_mean, S_logvar, m_logits = model.encode(_hist_batch(batch, L))
    S_post = model.reparam(S_mean, S_logvar, sample=True)      # [B,L,ds]
    m_post = F.gumbel_softmax(m_logits, tau=1.0, hard=False)   # [B,L,K]

    # ---- VFE over history: recon + transition KL + mental KL + posterior CE - #
    recon = 0.0
    for t in range(L):
        o_hat = model.decode(S_post[:, t])
        recon = recon + recon_loss(model, o_hat, recon_target(model, batch, t))
    recon = recon / L

    trans_kl = 0.0
    for t in range(1, L):
        pm, pl = model.transition(S_post[:, t - 1], batch["act_prev"][:, t])  # a_{t-1}
        trans_kl = trans_kl + gaussian_kl(S_mean[:, t], S_logvar[:, t], pm, pl)
    trans_kl = trans_kl / max(L - 1, 1) + 1e-3 * std_normal_kl(S_mean[:, 0], S_logvar[:, 0])

    mental_kl = 0.0
    for t in range(1, L):
        pm_logits = model.mental_transition(m_post[:, t - 1], S_post[:, t])
        mental_kl = mental_kl + cat_kl(m_logits[:, t], pm_logits)
    mental_kl = mental_kl / max(L - 1, 1)

    post_ce = 0.0
    for t in range(L):
        pl = model.policy_logits(S_post[:, t], m_post[:, t], batch["act_prev"][:, t])
        ce = F.cross_entropy(pl, batch["act"][:, t], weight=model.class_weight, reduction="none")
        chg = (batch["act"][:, t] != batch["act_prev"][:, t]).float()  # change-point
        post_ce = post_ce + ce * (1.0 + cfg.chg_weight * chg)
    post_ce = post_ce / L

    vfe = cfg.recon_w * recon + cfg.beta_kl * trans_kl + cfg.act_w * post_ce
    metrics = dict(recon=recon.mean().item(), trans_kl=trans_kl.mean().item(),
                   mental_kl=mental_kl.mean().item(), post_ce=post_ce.mean().item())

    if stage == "pretrain":
        loss = vfe.mean()
        metrics["vfe"] = vfe.mean().item()
        return loss, metrics

    # ---- EFE term: prior rollout imitation over future window (Eq. 3 use) --- #
    S = S_mean[:, L - 1]
    m = m_post[:, L - 1]
    a_prev = batch["act"][:, L - 1]                    # bootstrap action a_{L-1}
    roll_ce, roll_recon = 0.0, 0.0
    for j in range(P):
        pm, pl = model.transition(S, a_prev)
        S = model.reparam(pm, pl, sample=True)
        m_logits_j = model.mental_transition(m, S)
        m = F.gumbel_softmax(m_logits_j, tau=1.0, hard=False)
        logits = model.policy_logits(S, m, a_prev)
        tgt = batch["act"][:, L + j]
        ce = F.cross_entropy(logits, tgt, weight=model.class_weight, reduction="none")
        chg = (tgt != a_prev).float()
        roll_ce = roll_ce + ce * (1.0 + cfg.chg_weight * chg)
        o_hat = model.decode(S)
        roll_recon = roll_recon + recon_loss(model, o_hat, recon_target(model, batch, L + j))
        a_prev = tgt                                   # teacher forcing
    roll_ce = roll_ce / P
    roll_recon = roll_recon / P
    efe = roll_ce + 0.1 * roll_recon

    # load-balancing on discrete mental state to avoid collapse:
    # KL( batch-mean q(m) || uniform ), encourages diverse mode usage.
    K = model.cfg.n_mental
    mbar = m_post.mean(dim=(0, 1)).clamp_min(1e-8)
    balance = (mbar * (mbar.log() + math.log(K))).sum()

    loss = (vfe + cfg.eta * efe + cfg.gamma * mental_kl).mean() + 0.02 * balance
    if "m_target" in batch and model.cfg.n_mental > 1:  # DDXPlus supervised phase
        mt = batch["m_target"].clamp(0, model.cfg.n_mental - 1)
        msup = 0.0
        for t in range(L):
            msup = msup + F.cross_entropy(m_logits[:, t], mt[:, t], reduction="none")
        loss = loss + 0.5 * (msup / L).mean()
    metrics.update(efe=efe.mean().item(), roll_ce=roll_ce.mean().item(),
                   vfe=vfe.mean().item(), loss=loss.item())
    return loss, metrics


# --------------------------------------------------------------------------- #
#  Inference: EFE planning + hybrid rule/plan fusion                           #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def efe_scores(model: WorldModel, S, m_id, cfg: TrainConfig, a_prev=None):
    """Expected free energy per first-action via short MAP rollout (beam-lite).

    EFE(a) = -log pi(a|S,m) + lookahead_cost(a)   (lower = better)
    lookahead accumulates pragmatic (expected surprise of preferred outcome)
    and epistemic (predictive entropy) terms over horizon H.
    """
    A = model.cfg.n_actions
    B = S.shape[0]
    base = -model.policy_logits(S, m_id, a_prev).log_softmax(-1)  # [B,A]
    Sr = S.repeat_interleave(A, 0)
    mr = m_id.repeat_interleave(A, 0)
    a_cur = torch.arange(A, device=S.device).repeat(B)
    look = torch.zeros(B * A, device=S.device)
    for h in range(cfg.horizon):
        pm, pl = model.transition(Sr, a_cur)
        Sr = pm
        m_logits = model.mental_transition(mr, Sr)
        mr = m_logits.argmax(-1)
        p = model.policy_logits(Sr, mr, a_cur).softmax(-1)
        ent = -(p * p.clamp_min(1e-8).log()).sum(-1)
        prag = -p.max(-1).values.clamp_min(1e-8).log()
        look = look + prag + 0.1 * ent
        a_cur = p.argmax(-1)
    return base + cfg.plan_look_w * look.view(B, A)


@torch.no_grad()
def hybrid_step(model: WorldModel, rules: Optional[RuleLibrary], S, m_id, cfg: TrainConfig,
                use_rules=True, use_plan=True, greedy_rule=False, a_prev=None):
    """One-step hybrid action distribution (Eq. 5):
        p(a) ∝ pi_rule(a) + (1 - 1_hit) * exp(-EFE(a)/tau)
    Returns (prob [B,A], hit [B])."""
    B = S.shape[0]
    A = model.cfg.n_actions
    hit = torch.zeros(B, dtype=torch.bool, device=S.device)
    vote = torch.zeros(B, A, device=S.device)
    if use_rules and rules is not None and len(rules) > 0:
        vote, hit, _ = rules.activate(S, m_id)
    if greedy_rule:
        # greedy rule selection: if hit, use the rule vote directly (no planning)
        plan = model.policy_logits(S, m_id, a_prev).softmax(-1)
        vote_norm = vote / vote.sum(-1, keepdim=True).clamp_min(1e-8)
        prob = torch.where(hit.unsqueeze(1) & (vote.sum(-1, keepdim=True) > 0), vote_norm, plan)
        return prob, hit
    # Rules bypass planning (Sec. 4.1): when *every* sample in the (batch-size-1
    # at inference) decision has a confident rule hit, skip the costly EFE
    # beam-search rollout and act from the cheap policy head + rule vote.  This
    # is where the rule library speeds up inference; non-hit decisions still pay
    # for full planning.
    if use_plan and not (hit.numel() and bool(hit.all())):
        efe = efe_scores(model, S, m_id, cfg, a_prev)
        plan = (-efe / max(cfg.plan_temp, 1e-3)).softmax(-1)
    else:
        plan = model.policy_logits(S, m_id, a_prev).softmax(-1)
    # Fuse (Eq. 5) as a *bounded* soft blend: the rule prior dominates only in
    # proportion to how confidently a rule matches (top kappa*rho), capped so the
    # planning distribution always retains >= (1-alpha_max) weight.  This makes
    # rule activation robust to the kernel bandwidth -- confident, reliable rules
    # boost their action; spurious weak matches cannot degrade a good policy.
    v_sum = vote.sum(-1, keepdim=True)
    rule_dist = vote / v_sum.clamp_min(1e-8)
    alpha_max = getattr(rules.cfg, "fusion_alpha", 0.2) if rules is not None else 0.2
    rule_conf = vote.max(-1, keepdim=True).values.clamp(0.0, 1.0)     # top kappa*rho
    # Rules take over only where the plan is genuinely *ambiguous* (small top1-top2
    # margin -- the rare / edge cases the paper targets) and a confident rule
    # matches.  When the plan decisively prefers an action (large margin), rules
    # have zero influence, so they never degrade confident dominant-class
    # predictions; they only sharpen decisions the world model is unsure about.
    top2 = plan.topk(2, dim=-1).values
    margin = (top2[:, :1] - top2[:, 1:2])                            # [B,1]
    gate = (1.0 - margin / 0.3).clamp(0.0, 1.0)                      # 0 when margin>=0.3
    alpha = alpha_max * rule_conf * gate
    alpha = torch.where(v_sum > 0, alpha, torch.zeros_like(alpha))
    prob = (1 - alpha) * plan + alpha * rule_dist
    prob = prob / prob.sum(-1, keepdim=True).clamp_min(1e-8)
    return prob, hit


@torch.no_grad()
def predict(model: WorldModel, rules: Optional[RuleLibrary], batch, cfg: TrainConfig,
            use_rules=True, use_plan=True, greedy_rule=False):
    """Autoregressive multi-step forecast. Returns dict with:
        preds [B,P] argmax actions, hits [B,P], latency_ms (per decision step)."""
    L, P = cfg.hist_len, cfg.pred_len
    model.eval()
    S_mean, S_logvar, m_logits = model.encode(_hist_batch(batch, L))
    S = S_mean[:, L - 1]
    m_id = m_logits[:, L - 1].argmax(-1)
    a_prev = batch["act"][:, L - 1]
    # optional repeat-action mask: a diagnostic agent does not re-ask an evidence.
    # Seed it with the actions already taken in the history window.
    mask = None
    thr = getattr(cfg, "mask_repeat_below", 0)
    if thr > 0:
        A = model.cfg.n_actions
        mask = torch.zeros(S.shape[0], A, dtype=torch.bool, device=S.device)
        hist_a = batch["act"][:, :L]
        mask.scatter_(1, hist_a.clamp(0, A - 1), True)
        mask[:, thr:] = False                          # only mask ASK ids < thr
    preds, hits = [], []
    t0 = time.perf_counter()
    for j in range(P):
        pm, _ = model.transition(S, a_prev)
        S = pm
        m_id = model.mental_transition(m_id, S).argmax(-1)
        prob, hit = hybrid_step(model, rules, S, m_id, cfg, use_rules, use_plan,
                                greedy_rule, a_prev=a_prev)
        if mask is not None:
            prob = prob.masked_fill(mask, 0.0)
        a = prob.argmax(-1)
        preds.append(a); hits.append(hit)
        if mask is not None:
            upd = a < thr
            mask[torch.arange(a.shape[0], device=a.device)[upd], a[upd]] = True
        a_prev = a                                     # autoregressive
    dt = (time.perf_counter() - t0) * 1000.0 / P       # ms per decision step (batch)
    preds = torch.stack(preds, 1)                      # [B,P]
    hits = torch.stack(hits, 1)
    return dict(preds=preds, hits=hits, latency_ms=dt / max(batch["act"].shape[0], 1) * batch["act"].shape[0])


@torch.no_grad()
def harvest_stats(model: WorldModel, batch, cfg: TrainConfig):
    """Collect (S^MAP, m^MAP, a, per-step free energy) at future decision steps
    for wake-phase rule growing (Alg. 1 lines 6-8)."""
    L, P = cfg.hist_len, cfg.pred_len
    S_mean, _, m_logits = model.encode(_hist_batch(batch, L))
    S = S_mean[:, L - 1]
    m_id = m_logits[:, L - 1].argmax(-1)
    a_prev = batch["act"][:, L - 1]
    Ss, Ms, As, Fs = [], [], [], []
    for j in range(P):
        pm, pl = model.transition(S, a_prev)
        S = pm
        m_id = model.mental_transition(m_id, S).argmax(-1)
        logits = model.policy_logits(S, m_id, a_prev)
        tgt = batch["act"][:, L + j]
        ce = F.cross_entropy(logits, tgt, reduction="none")
        o_hat = model.decode(S)
        rc = recon_loss(model, o_hat, recon_target(model, batch, L + j))
        fe = ce + 0.1 * rc
        Ss.append(S); Ms.append(m_id); As.append(tgt); Fs.append(fe)
        a_prev = tgt
    return (torch.cat(Ss), torch.cat(Ms), torch.cat(As), torch.cat(Fs))
