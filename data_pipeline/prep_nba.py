"""NBA SportVU preprocessing (Appendix C.2).

Extracts frame-level ball/player coordinates from SportVU logs, downsamples,
and builds per-player action trajectories with 7 basketball actions:
    0 straight, 1 left, 2 right, 3 turnaround, 4 dribble, 5 shoot, 6 pass
Actions follow the piecewise definition of Sec. C.2 (ball-proximity for
dribble/pass/shoot; heading change for straight/left/right/turnaround).
Per-step symbolic features (18-d): soft direction kernels to 3 nearest others
(x4 basis directions) + relational distances/angles [d1,d2,d3,d_rim,theta_rim,d_mean].
"""
import os, sys, glob, json, argparse, tempfile, shutil
import numpy as np
import multiprocessing as mp
import py7zr

COURT_W, COURT_H = 94.0, 50.0
BASKETS = np.array([[5.25, 25.0], [88.75, 25.0]])
D_DRIB, D_REL, D_REC = 3.0, 6.0, 3.0
SIGMA = 10.0
DS = 2                      # downsample stride over 25Hz moments (~12.5Hz)
MIN_RUN = 3                 # merge single-frame action flips (classification denoise)
HWIN = 2                    # frames for windowed heading (robust turn detection)
T = 15                     # window length (hist 10 + pred 5)
BASIS = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32)


SHOT_Z = 8.0        # ball height (ft) indicating a shot arc
HOLD = 4.0          # possession radius (ft)


def movement_action(pls, pi, t, n):
    # windowed heading: displacement over +/- HWIN frames (robust to noise,
    # so turns register across multiple output frames)
    t0 = max(0, t - HWIN); t1 = min(n - 1, t + HWIN)
    if t - HWIN < 1 or t + HWIN > n - 1:
        v_prev = pls[t, pi] - pls[t - 1, pi] if t >= 1 else np.zeros(2)
        v_cur = pls[min(t + 1, n - 1), pi] - pls[t, pi]
    else:
        v_prev = pls[t, pi] - pls[t0, pi]
        v_cur = pls[t1, pi] - pls[t, pi]
    sp = np.linalg.norm(v_cur)
    if sp < 0.5 or np.linalg.norm(v_prev) < 1e-3:
        return 0
    h0 = v_prev / (np.linalg.norm(v_prev) + 1e-6)
    h1 = v_cur / (sp + 1e-6)
    cross = h0[0] * h1[1] - h0[1] * h1[0]
    dot = float(np.clip(h0 @ h1, -1, 1))
    a = np.degrees(np.arctan2(cross, dot))
    if abs(a) < 15:
        return 0
    if abs(a) >= 55:
        return 3            # turnaround
    return 1 if a > 0 else 2  # left / right


def classify_and_featurize(moments):
    """moments: list of (game_clock, ball_xyz[3], players_xy[10,2]).
    Detects clip-level ball events (dribble/pass/shoot) from the ball-height and
    handler-change signal, then builds per-player action trajectories for the
    ball-involved players.  7 actions: straight/left/right/turnaround/dribble/shoot/pass.
    """
    n = len(moments)
    if n < T + 1:
        return []
    balls = np.array([m[1][:2] for m in moments], dtype=np.float32)     # [n,2]
    ballz = np.array([m[1][2] for m in moments], dtype=np.float32)      # [n]
    pls = np.array([m[2] for m in moments], dtype=np.float32)           # [n,10,2]
    d_ball = np.linalg.norm(pls - balls[:, None, :], axis=2)           # [n,10]
    handler = np.argmin(d_ball, axis=1)                                # [n]
    dmin = d_ball[np.arange(n), handler]
    poss = dmin <= HOLD

    # ---- clip-level events: per-frame label for the on-ball player ---------- #
    # event_lab[t] in {dribble, pass, shoot, none}; owner[t] = which player
    event = np.full(n, -1, dtype=np.int64)   # -1 none, else action id
    owner = np.full(n, -1, dtype=np.int64)
    for t in range(1, n):
        if poss[t] and poss[t - 1] and handler[t] == handler[t - 1]:
            event[t] = 4; owner[t] = handler[t]                       # dribble
        # shot: ball goes airborne shortly after a possession
        if ballz[t] > SHOT_Z and ballz[t - 1] <= SHOT_Z:
            # attribute to most recent possessor
            k = t - 1
            while k >= 0 and not poss[k]:
                k -= 1
            if k >= 0:
                event[k] = 5; owner[k] = handler[k]                   # shoot
        # pass: handler changes between two low-ball possession frames
        if poss[t] and poss[t - 1] and handler[t] != handler[t - 1] and ballz[t] < SHOT_Z:
            event[t - 1] = 6; owner[t - 1] = handler[t - 1]           # pass by prev owner
    involved = set(np.unique(handler[poss]).tolist())
    if not involved:
        involved = set(range(10))

    trajs = []
    for pi in involved:
        obs = np.zeros((n, 22), dtype=np.float32)
        act = np.zeros(n, dtype=np.int64)
        for t in range(n):
            p = pls[t, pi]
            others = np.delete(pls[t], pi, axis=0)
            dj = np.linalg.norm(others - p, axis=1)
            order = np.argsort(dj)
            near3 = others[order[:3]]
            d3 = dj[order[:3]]
            feat_dir = []
            for k in range(3):
                yk = near3[k] if k < len(near3) else p
                u = yk - p
                nrm = np.linalg.norm(u) + 1e-6
                w = np.exp(-(nrm ** 2) / (2 * SIGMA ** 2))
                for bdir in BASIS:
                    feat_dir.append(w * max(float(bdir @ (u / nrm)), 0.0))
            db = np.linalg.norm(BASKETS - p, axis=1)
            ri = int(np.argmin(db))
            th_rim = np.arctan2(BASKETS[ri, 1] - p[1], BASKETS[ri, 0] - p[0])
            d1 = d3[0] if len(d3) > 0 else 0.0
            d2 = d3[1] if len(d3) > 1 else d1
            d3v = d3[2] if len(d3) > 2 else d2
            rel = [d1 / 47, d2 / 47, d3v / 47, db[ri] / 47, th_rim / np.pi, float(dj.mean()) / 47]
            # ball-relative features: distance, direction, approach rate
            bvec = balls[t] - p
            bd = np.linalg.norm(bvec) + 1e-6
            bdir = bvec / bd
            approach = (d_ball[t - 1, pi] - d_ball[t, pi]) if t >= 1 else 0.0
            ball_feat = [min(bd / 47, 2.0), float(bdir[0]), float(bdir[1]), float(np.clip(approach / 5.0, -2, 2))]
            obs[t] = np.array(feat_dir + rel + ball_feat, dtype=np.float32)
            if t == 0:
                act[t] = 0
            elif owner[t] == pi and event[t] >= 0:
                act[t] = event[t]                                     # dribble/shoot/pass
            else:
                act[t] = movement_action(pls, pi, t, n)
        act = smooth_actions(act, MIN_RUN)
        for s in range(1, n - T + 1, T):
            trajs.append((obs[s:s + T], act[s:s + T]))
    return trajs


def smooth_actions(a, min_run=3):
    """Merge action runs shorter than min_run into the previous run (removes
    single-frame classification flips)."""
    a = a.copy()
    changed = True
    while changed:
        changed = False
        bounds = [0] + list(np.where(a[1:] != a[:-1])[0] + 1) + [len(a)]
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if e - s < min_run and s > 0:
                a[s:e] = a[s - 1]
                changed = True
                break
    return a


def parse_game(path):
    tmp = tempfile.mkdtemp()
    try:
        with py7zr.SevenZipFile(path, "r") as z:
            z.extractall(path=tmp)
        js = glob.glob(os.path.join(tmp, "*.json"))
        if not js:
            return []
        d = json.load(open(js[0]))
        out = []
        events = d["events"]
        # sample up to 20 clips (events) per game, evenly
        idxs = np.linspace(0, len(events) - 1, min(20, len(events))).astype(int)
        for ei in idxs:
            moments = events[ei]["moments"]
            seq = []
            last_gc = None
            for mi, mom in enumerate(moments):
                if mi % DS != 0:
                    continue
                gc = mom[2]
                pos = mom[5]
                if pos is None or len(pos) != 11:
                    continue
                ball = np.array(pos[0][2:5], dtype=np.float32)   # x,y,z
                players = np.array([pp[2:4] for pp in pos[1:11]], dtype=np.float32)
                if players.shape != (10, 2) or ball.shape != (3,):
                    continue
                if last_gc is not None and gc == last_gc:
                    continue
                last_gc = gc
                seq.append((gc, ball, players))
            out.extend(classify_and_featurize(seq))
        return out
    except Exception as e:
        return []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data_raw/nba_repo/data/2016.NBA.Raw.SportVU.Game.Logs")
    ap.add_argument("--out", default="../data_proc/nba")
    ap.add_argument("--n_train", type=int, default=9840)
    ap.add_argument("--n_val", type=int, default=2460)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    games = sorted(glob.glob(os.path.join(args.raw, "*.7z")))
    print(f"{len(games)} games")
    n_val_games = max(1, int(len(games) * 0.2))
    val_games = games[-n_val_games:]
    train_games = games[:-n_val_games]

    def collect(glist):
        obs_all, act_all = [], []
        with mp.Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(parse_game, glist)):
                for o, a in res:
                    obs_all.append(o); act_all.append(a)
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(glist)} games, {len(obs_all)} windows")
        return np.array(obs_all, dtype=np.float32), np.array(act_all, dtype=np.int64)

    rng = np.random.default_rng(0)
    print("processing train games...")
    obs, act = collect(train_games)
    if len(obs) > args.n_train:
        idx = rng.choice(len(obs), args.n_train, replace=False); obs, act = obs[idx], act[idx]
    np.savez(os.path.join(args.out, "train.npz"), obs=obs, act=act)
    print("train", obs.shape, "dist%", np.round(np.bincount(act.ravel(), minlength=7) / act.size * 100, 1))

    print("processing val games...")
    obs, act = collect(val_games)
    if len(obs) > args.n_val:
        idx = rng.choice(len(obs), args.n_val, replace=False); obs, act = obs[idx], act[idx]
    np.savez(os.path.join(args.out, "val.npz"), obs=obs, act=act)
    print("val", obs.shape, "dist%", np.round(np.bincount(act.ravel(), minlength=7) / act.size * 100, 1))


if __name__ == "__main__":
    main()
