"""Atari-Berzerk preprocessing (Appendix C.5) from the Atari-HEAD dataset.

Each trajectory is a .txt (frame_id,...,action,gaze) plus a .tar.bz2 of PNG
frames.  Frames are converted to 128x128 grayscale; each is paired with its
human action (18 discrete).  Sequences are segmented into fixed windows and the
frames stored in a uint8 memmap referenced by per-window frame indices.
"""
import os, sys, glob, io, argparse, tarfile, re
import numpy as np
import multiprocessing as mp
from PIL import Image

IMG = 128
DS = 5                 # frame downsample (Atari-HEAD ~ per-frame human play)
MIN_RUN = 3            # merge very short action flips (input-jitter denoise)
T = 15                 # window (hist 10 + pred 5)
N_ACT = 18


def smooth_actions(a, min_run=3):
    a = a.copy()
    changed = True
    while changed:
        changed = False
        import numpy as _np
        bounds = [0] + list(_np.where(a[1:] != a[:-1])[0] + 1) + [len(a)]
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if e - s < min_run and s > 0:
                a[s:e] = a[s - 1]; changed = True; break
    return a


def parse_actions(txt):
    """Return dict frame_number -> action id."""
    acts = {}
    with open(txt) as f:
        next(f)  # header
        for line in f:
            parts = line.split(",", 7)
            if len(parts) < 6:
                continue
            fid = parts[0]                      # e.g. RZ_9934643_123
            m = re.search(r"_(\d+)$", fid)
            if not m:
                continue
            try:
                a = int(parts[5])
            except ValueError:
                continue
            acts[int(m.group(1))] = min(max(a, 0), N_ACT - 1)
    return acts


def process_traj(args):
    txt, tar = args
    acts = parse_actions(txt)
    if not acts:
        return None
    frames_ns = sorted(acts.keys())
    keep = set(frames_ns[::DS])
    imgs = {}
    try:
        with tarfile.open(tar, "r:bz2") as tf:
            for m in tf:
                if not m.name.endswith(".png"):
                    continue
                mm = re.search(r"_(\d+)\.png$", m.name)
                if not mm:
                    continue
                n = int(mm.group(1))
                if n not in keep:
                    continue
                fobj = tf.extractfile(m)
                if fobj is None:
                    continue
                im = Image.open(io.BytesIO(fobj.read())).convert("L").resize((IMG, IMG))
                imgs[n] = np.asarray(im, dtype=np.uint8)
    except Exception as e:
        return None
    seq_ns = [n for n in frames_ns[::DS] if n in imgs]
    if len(seq_ns) < T + 1:
        return None
    frames = np.stack([imgs[n] for n in seq_ns])          # [nf,128,128]
    act = np.array([acts[n] for n in seq_ns], dtype=np.int64)
    act = smooth_actions(act, MIN_RUN)
    return frames, act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data_raw/atari_ext/berzerk")
    ap.add_argument("--out", default="../data_proc/atari_berzerk")
    ap.add_argument("--n_train", type=int, default=16541)
    ap.add_argument("--n_val", type=int, default=16581)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    txts = sorted(glob.glob(os.path.join(args.raw, "*.txt")))
    pairs = []
    for t in txts:
        tar = t[:-4] + ".tar.bz2"
        if os.path.exists(tar):
            pairs.append((t, tar))
    print(f"{len(pairs)} trajectories")
    n_val_traj = max(1, len(pairs) // 2)
    splits = {"val": pairs[:n_val_traj], "train": pairs[n_val_traj:]}

    for split, plist in splits.items():
        n_max = args.n_train if split == "train" else args.n_val
        all_frames, win_idx, win_act = [], [], []
        offset = 0
        with mp.Pool(args.workers) as pool:
            for res in pool.imap_unordered(process_traj, plist):
                if res is None:
                    continue
                frames, act = res
                nf = len(frames)
                all_frames.append(frames)
                for s in range(0, nf - T + 1, args.stride):
                    win_idx.append(np.arange(offset + s, offset + s + T))
                    win_act.append(act[s:s + T])
                offset += nf
                print(f"  [{split}] traj frames={nf} total_windows={len(win_idx)}")
        frames = np.concatenate(all_frames)                # [M,128,128]
        win_idx = np.array(win_idx, dtype=np.int64)
        win_act = np.array(win_act, dtype=np.int64)
        rng = np.random.default_rng(0 if split == "train" else 1)
        if len(win_idx) > n_max:
            sel = rng.choice(len(win_idx), n_max, replace=False)
            win_idx, win_act = win_idx[sel], win_act[sel]
        M = frames.shape[0]
        mm_path = os.path.join(args.out, f"{split}_frames.u8")
        mm = np.memmap(mm_path, dtype=np.uint8, mode="w+", shape=(M, IMG, IMG))
        mm[:] = frames; mm.flush()
        np.savez(os.path.join(args.out, f"{split}_meta.npz"),
                 act=win_act, frame_idx=win_idx, memmap_path=os.path.abspath(mm_path), M=M, H=IMG, W=IMG)
        print(f"[{split}] frames={M} windows={len(win_act)} "
              f"dist%={np.round(np.bincount(win_act.ravel(), minlength=N_ACT)/win_act.size*100,1)}")


if __name__ == "__main__":
    main()
