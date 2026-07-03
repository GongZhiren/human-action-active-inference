"""Car-Following preprocessing (Appendix C.3).

Reads the per-timestamp regime sequences (RomainLITUD HV-vs-AV) at 10 Hz and
builds fixed windows of length T = hist+pred.  Each step's observation packs
[run_len, since_last, recent_change_rate]; the action is the driving regime.
Seven regimes: F, A, D, Fd, Fa, C, S.
"""
import os, sys, argparse
import numpy as np
import pandas as pd

REGIMES = ["F", "A", "D", "Fd", "Fa", "C", "S"]
RID = {r: i for i, r in enumerate(REGIMES)}


def smooth_runs(r, min_run=5):
    """Merge runs shorter than min_run into the previous run (removes sub-0.5s
    spurious regime flips; regimes are behavior segments lasting seconds)."""
    r = r.copy()
    changed = True
    while changed:
        changed = False
        # find run boundaries
        bounds = [0] + list(np.where(r[1:] != r[:-1])[0] + 1) + [len(r)]
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if e - s < min_run:
                fill = r[s - 1] if s > 0 else (r[e] if e < len(r) else r[s])
                r[s:e] = fill
                changed = True
                break
    return r


def build_windows(seqs, T, stride, max_windows, seed=0, min_run=10):
    rng = np.random.default_rng(seed)
    obs_list, act_list = [], []
    for r in seqs:
        r = smooth_runs(r, min_run)
        n = len(r)
        if n < T:
            continue
        # run length + since-last-change features
        run = np.ones(n, dtype=np.float32)
        for t in range(1, n):
            run[t] = run[t - 1] + 1 if r[t] == r[t - 1] else 1.0
        change = (np.concatenate([[0], (r[1:] != r[:-1]).astype(np.float32)]))
        chg_rate = np.convolve(change, np.ones(10) / 10.0, mode="same").astype(np.float32)
        since = run * 0.1
        feat = np.stack([np.clip(run / 50.0, 0, 2), np.clip(since / 5.0, 0, 2), chg_rate], 1)
        for s in range(0, n - T + 1, stride):
            obs_list.append(feat[s:s + T])
            act_list.append(r[s:s + T])
    obs = np.asarray(obs_list, dtype=np.float32)
    act = np.asarray(act_list, dtype=np.int64)
    if len(obs) > max_windows:
        idx = rng.choice(len(obs), max_windows, replace=False)
        obs, act = obs[idx], act[idx]
    return obs, act


def load_regime_seqs(h5_files, limit_cases=None):
    seqs = []
    for f in h5_files:
        df = pd.read_hdf(f)
        for cid, grp in df.groupby("case_id", sort=False):
            r = grp["regime"].map(RID).values.astype(np.int64)
            seqs.append(r)
            if limit_cases and len(seqs) >= limit_cases:
                break
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data_raw/car_repo/ext/regimes")
    ap.add_argument("--out", default="../data_proc/car")
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--n_train", type=int, default=19250)
    ap.add_argument("--n_val", type=int, default=2500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_files = [os.path.join(args.raw, f"regimes_all_{p}_train.h5") for p in ("HH", "HA", "AH")]
    val_files = [os.path.join(args.raw, f"regimes_all_{p}_val.h5") for p in ("HH", "HA", "AH")]
    train_files = [f for f in train_files if os.path.exists(f)]
    val_files = [f for f in val_files if os.path.exists(f)]

    print("loading train regime sequences...")
    tr_seqs = load_regime_seqs(train_files, limit_cases=8000)
    print(f"  {len(tr_seqs)} cases")
    obs, act = build_windows(tr_seqs, args.T, args.stride, args.n_train, seed=1)
    np.savez(os.path.join(args.out, "train.npz"), obs=obs, act=act)
    print("train", obs.shape, act.shape, "dist", np.round(np.bincount(act.ravel(), minlength=7) / act.size * 100, 1))

    print("loading val regime sequences...")
    va_seqs = load_regime_seqs(val_files, limit_cases=3000)
    obs, act = build_windows(va_seqs, args.T, args.stride, args.n_val, seed=2)
    np.savez(os.path.join(args.out, "val.npz"), obs=obs, act=act)
    print("val", obs.shape, act.shape, "dist", np.round(np.bincount(act.ravel(), minlength=7) / act.size * 100, 1))


if __name__ == "__main__":
    main()
