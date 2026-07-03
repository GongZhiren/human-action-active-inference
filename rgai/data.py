"""Unified dataset interface.

Every split is stored as an .npz with:
    act   : int64 [N, T]                 (action id at each step; T = hist_len+pred_len)
    obs   : float32 [N, T, F]            (numeric backbones)          -- optional
    tok   : int64 [N, T]                 (evidence2vec token ids)     -- optional
    extra : float32 [N, T, E]            (evidence2vec side features) -- optional
    mstate: int64 [N, T]                 (supervised mental state)    -- optional
Vision (Atari) frames are stored in a uint8 memmap; see FrameDataset.
"""
from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    def __init__(self, npz_path, n_actions):
        d = np.load(npz_path, allow_pickle=True)
        self.act = d["act"].astype(np.int64)
        self.obs = d["obs"].astype(np.float32) if "obs" in d.files else None
        self.tok = d["tok"].astype(np.int64) if "tok" in d.files else None
        self.extra = d["extra"].astype(np.float32) if "extra" in d.files else None
        self.mstate = d["mstate"].astype(np.int64) if "mstate" in d.files else None
        self.n_actions = n_actions
        self.T = self.act.shape[1]

    def __len__(self):
        return self.act.shape[0]

    def __getitem__(self, i):
        item = {"act": self.act[i]}
        if self.obs is not None:
            item["obs"] = self.obs[i]
        if self.tok is not None:
            item["tok"] = self.tok[i]
            item["extra"] = self.extra[i]
        if self.mstate is not None:
            item["mstate"] = self.mstate[i]
        return item

    def collate(self, batch):
        out = {}
        act = torch.tensor(np.stack([b["act"] for b in batch]), dtype=torch.long)
        out["act"] = act
        pad = self.n_actions
        act_prev = torch.full_like(act, pad)
        act_prev[:, 1:] = act[:, :-1]
        out["act_prev"] = act_prev
        if "obs" in batch[0]:
            out["obs"] = torch.tensor(np.stack([b["obs"] for b in batch]), dtype=torch.float32)
        if "tok" in batch[0]:
            out["tok"] = torch.tensor(np.stack([b["tok"] for b in batch]), dtype=torch.long)
            out["extra"] = torch.tensor(np.stack([b["extra"] for b in batch]), dtype=torch.float32)
        if "mstate" in batch[0]:
            out["m_target"] = torch.tensor(np.stack([b["mstate"] for b in batch]), dtype=torch.long)
        return out


class FrameDataset(Dataset):
    """Atari: frames stored as a uint8 memmap [M, H, W]; index arrays map each
    sample's T steps into the memmap.  act aligned per step."""
    def __init__(self, meta_path, n_actions, img_size=128):
        d = np.load(meta_path, allow_pickle=True)
        self.act = d["act"].astype(np.int64)          # [N,T]
        self.frame_idx = d["frame_idx"].astype(np.int64)  # [N,T] -> row in memmap
        self.mstate = d["mstate"].astype(np.int64) if "mstate" in d.files else None
        mp = str(d["memmap_path"])
        if not os.path.isabs(mp):
            # resolve relative to the meta file's directory
            mp = os.path.join(os.path.dirname(os.path.abspath(meta_path)), os.path.basename(mp))
        self.memmap_path = mp
        self.M = int(d["M"]); self.H = int(d["H"]); self.W = int(d["W"])
        self.n_actions = n_actions
        self.T = self.act.shape[1]
        self._mm = None

    @property
    def mm(self):
        if self._mm is None:
            self._mm = np.memmap(self.memmap_path, dtype=np.uint8, mode="r",
                                 shape=(self.M, self.H, self.W))
        return self._mm

    def __len__(self):
        return self.act.shape[0]

    def __getitem__(self, i):
        idx = self.frame_idx[i]
        frames = np.asarray(self.mm[idx]).astype(np.float32) / 255.0  # [T,H,W]
        item = {"act": self.act[i], "frames": frames[:, None]}        # [T,1,H,W]
        if self.mstate is not None:
            item["mstate"] = self.mstate[i]
        return item

    def collate(self, batch):
        out = {}
        act = torch.tensor(np.stack([b["act"] for b in batch]), dtype=torch.long)
        out["act"] = act
        pad = self.n_actions
        act_prev = torch.full_like(act, pad)
        act_prev[:, 1:] = act[:, :-1]
        out["act_prev"] = act_prev
        out["frames"] = torch.tensor(np.stack([b["frames"] for b in batch]), dtype=torch.float32)
        if "mstate" in batch[0]:
            out["m_target"] = torch.tensor(np.stack([b["mstate"] for b in batch]), dtype=torch.long)
        return out
