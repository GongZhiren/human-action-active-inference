"""Neural components of the RGAI world model.

Implements the generative factorization of Eq. (4):
    p_phi(O_t | S_t)                     -- decoder / likelihood
    p_phi(S_t | S_{t-1}, a_{t-1})        -- state transition prior
    p_phi(m_t | m_{t-1}, S_t)            -- (sticky) mental-state transition
    p_phi_pi(a_t | S_t, m_t)             -- rule-augmentable policy head
and the amortized recognition network q_theta(S_t, m_t | H_t) (encoder).
"""
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def mlp(sizes, act=nn.GELU, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
#  Sequence backbones producing per-step hidden features from history          #
# --------------------------------------------------------------------------- #
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.inp = nn.Linear(cfg.obs_dim, d)
        self.act_emb = nn.Embedding(cfg.n_actions + 1, d)  # +1 for BOS/pad
        self.pos = PositionalEncoding(d)
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, dim_feedforward=4 * d, dropout=cfg.dropout,
            batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, cfg.n_layers)
        self.out_dim = d

    def forward(self, obs, act):
        # obs [B,L,obs_dim], act [B,L] (previous actions, id in [0,n_actions], n_actions=pad)
        h = self.inp(obs) + self.act_emb(act)
        h = self.pos(h)
        return self.enc(h)


class BiGRUBackbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.hidden_dim
        self.inp = nn.Linear(cfg.obs_dim, d)
        self.act_emb = nn.Embedding(cfg.n_actions + 1, d)
        self.gru = nn.GRU(d, d // 2, cfg.n_layers, batch_first=True,
                          bidirectional=True, dropout=cfg.dropout if cfg.n_layers > 1 else 0.0)
        self.out_dim = d

    def forward(self, obs, act):
        h = self.inp(obs) + self.act_emb(act)
        out, _ = self.gru(h)
        return out


class CNNEncoder(nn.Module):
    """Frame encoder for Atari: 128x128 grayscale -> feature vector."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.img_channels
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4, 2), nn.GELU(),     # 128->32
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),    # 32->16
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),   # 16->8
            nn.Conv2d(128, 128, 4, 2, 1), nn.GELU(),  # 8->4
        )
        self.fc = nn.Linear(128 * 4 * 4, cfg.hidden_dim)
        self.out_dim = cfg.hidden_dim

    def forward(self, frames):
        # frames [B,L,C,H,W]
        B, L = frames.shape[:2]
        x = frames.reshape(B * L, *frames.shape[2:])
        z = self.net(x).reshape(B * L, -1)
        z = self.fc(z).reshape(B, L, -1)
        return z


class CNNDecoder(nn.Module):
    """Reconstruct 128x128 grayscale frame from continuous latent S."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.state_dim, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.GELU(),  # 4->8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),   # 8->16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),    # 16->32
            nn.ConvTranspose2d(32, cfg.img_channels, 8, 4, 2), # 32->128
        )

    def forward(self, S):
        # returns raw logits (sigmoid applied in loss via BCE-with-logits, AMP-safe)
        x = self.fc(S).reshape(-1, 128, 4, 4)
        return self.net(x)


class Evidence2Vec(nn.Module):
    """DDXPlus: embed evidence tokens and build the state representation from
    (evidence embedding, top-K posterior over diagnoses, posterior entropy)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size + 1, cfg.hidden_dim, padding_idx=cfg.vocab_size)
        self.proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.extra_proj = nn.Linear(max(cfg.extra_dim, 1), cfg.hidden_dim)
        self.out_dim = cfg.hidden_dim

    def forward(self, tok, extra):
        # tok [B,L] evidence-id ; extra [B,L,extra_dim] (topk posterior + entropy)
        return self.proj(self.emb(tok)) + self.extra_proj(extra)


# --------------------------------------------------------------------------- #
#  World model                                                                 #
# --------------------------------------------------------------------------- #
class WorldModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        ds, dh, K = cfg.state_dim, cfg.hidden_dim, cfg.n_mental

        if cfg.backbone == "transformer":
            self.backbone = TransformerBackbone(cfg)
        elif cfg.backbone == "bigru":
            self.backbone = BiGRUBackbone(cfg)
        elif cfg.backbone == "cnn":
            self.frame_enc = CNNEncoder(cfg)
            tlayer = nn.TransformerEncoderLayer(
                dh, cfg.n_heads, 4 * dh, cfg.dropout, batch_first=True, activation="gelu")
            self.temporal = nn.TransformerEncoder(tlayer, cfg.n_layers)
            self.act_emb_v = nn.Embedding(cfg.n_actions + 1, dh)
            self.pos_v = PositionalEncoding(dh)
            self.decoder = CNNDecoder(cfg)
        elif cfg.backbone == "evidence2vec":
            self.ev = Evidence2Vec(cfg)
            tlayer = nn.TransformerEncoderLayer(
                dh, cfg.n_heads, 4 * dh, cfg.dropout, batch_first=True, activation="gelu")
            self.temporal = nn.TransformerEncoder(tlayer, cfg.n_layers)
            self.act_emb_v = nn.Embedding(cfg.n_actions + 1, dh)
            self.pos_v = PositionalEncoding(dh)
        else:
            raise ValueError(cfg.backbone)

        bdim = dh
        # recognition heads q_theta(S,m | H)
        self.q_S = mlp([bdim, dh, 2 * ds])          # -> mean, logvar
        self.q_m = mlp([bdim, dh, K])               # -> categorical logits
        # decoder p(O|S) for non-vision backbones (evidence2vec reconstructs the
        # numeric side-features 'extra'; others reconstruct the obs vector)
        if cfg.backbone == "evidence2vec":
            self.decoder = mlp([ds, dh, max(cfg.extra_dim, 1)])
        elif cfg.backbone != "cnn":
            self.decoder = mlp([ds, dh, cfg.obs_dim])
        # transition p(S'|S,a)
        self.act_emb_t = nn.Embedding(cfg.n_actions + 1, dh)
        self.trans = mlp([ds + dh, dh, 2 * ds])
        # mental transition p(m'|m,S') with stickiness
        self.m_emb = nn.Embedding(K, dh)
        self.m_trans = mlp([dh + ds, dh, K])
        self.m_stick = nn.Parameter(torch.tensor(2.0))  # bias to stay in same mode
        # policy head p(a|S,m,a_prev) -- also conditions on the previous action
        # (the paper's "previous-K one-hot regimes" history), enabling habitual
        # persistence and learned change-points.
        self.act_emb_p = nn.Embedding(cfg.n_actions + 1, dh)
        self.policy = mlp([ds + dh + dh, dh, cfg.n_actions])
        # optional inverse-frequency class weighting for the action CE (HHAR)
        self.register_buffer("class_weight", torch.ones(cfg.n_actions))

    # --- encoding ---------------------------------------------------------- #
    def encode(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if cfg.backbone == "cnn":
            z = self.frame_enc(batch["frames"])              # [B,L,dh]
            z = z + self.act_emb_v(batch["act_prev"])
            h = self.temporal(self.pos_v(z))
        elif cfg.backbone == "evidence2vec":
            e = self.ev(batch["tok"], batch["extra"])        # [B,L,dh]
            z = e + self.act_emb_v(batch["act_prev"])
            h = self.temporal(self.pos_v(z))
        else:
            h = self.backbone(batch["obs"], batch["act_prev"])
        S_stat = self.q_S(h)
        S_mean, S_logvar = S_stat.chunk(2, dim=-1)
        S_logvar = torch.clamp(S_logvar, -8, 4)
        m_logits = self.q_m(h)
        return S_mean, S_logvar, m_logits

    # --- generative pieces ------------------------------------------------- #
    def decode(self, S):
        return self.decoder(S)

    def transition(self, S, a):
        # residual transition: next state stays close to current state by default
        # (preserves encoded run/context so habitual persistence is easy; the
        # network learns the delta that changes the state when dynamics warrant).
        h = torch.cat([S, self.act_emb_t(a)], dim=-1)
        stat = self.trans(h)
        delta, logvar = stat.chunk(2, dim=-1)
        return S + delta, torch.clamp(logvar, -8, 4)

    def mental_transition(self, m_prev, S):
        # m_prev: [B] ids or [B,K] soft ; returns logits [B,K]
        if m_prev.dim() == 1:
            me = self.m_emb(m_prev)
            prev_oh = F.one_hot(m_prev, self.cfg.n_mental).float()
        else:
            me = m_prev @ self.m_emb.weight
            prev_oh = m_prev
        logits = self.m_trans(torch.cat([me, S], dim=-1))
        logits = logits + self.m_stick * prev_oh  # sticky prior toward staying
        return logits

    def policy_logits(self, S, m, a_prev=None):
        # m: [B] ids or [B,K] soft ; a_prev: [B] previous action ids (optional)
        if m.dim() == 1:
            me = self.m_emb(m)
        else:
            me = m @ self.m_emb.weight
        if a_prev is None:
            a_prev = torch.full((S.shape[0],), self.cfg.n_actions, dtype=torch.long, device=S.device)
        ap = self.act_emb_p(a_prev)
        return self.policy(torch.cat([S, me, ap], dim=-1))

    @staticmethod
    def reparam(mean, logvar, sample=True):
        if not sample:
            return mean
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
