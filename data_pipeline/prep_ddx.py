"""DDXPlus preprocessing (Appendix C.4).

Restricts to the URTI pathology.  Each patient becomes a trajectory of ASK
actions (querying present evidences) followed by a final DIAG action.  The
Evidence2Vec state at each step packs [evidence token id | top-K Naive-Bayes
posterior over diagnoses | posterior entropy].  The mental state m is the
diagnostic phase (1..5) derived from trajectory progress.
"""
import os, sys, ast, json, argparse
import numpy as np
import pandas as pd

TARGET = "URTI"
SHUFFLE_FRAC = 0.04   # perturb the canonical asking order (0=sorted/trivial, 1=original/hard)
TOPK = 8
T = 15          # window (hist 10 + pred 5)
N_PHASE = 5


def base_code(ev):
    return ev.split("_@_")[0]


def build_nb(df, conditions, evid_codes):
    """Bernoulli Naive-Bayes P(evidence present | condition)."""
    cidx = {c: i for i, c in enumerate(conditions)}
    eidx = {e: i for i, e in enumerate(evid_codes)}
    nC, nE = len(conditions), len(evid_codes)
    count = np.ones((nC, nE)) * 0.5      # Laplace smoothing
    ctot = np.ones(nC) * 1.0
    prior = np.ones(nC)
    # sample up to 300k patients for the NB teacher (fast, sufficient)
    if len(df) > 300000:
        df = df.sample(300000, random_state=0)
    for c, evstr in zip(df["PATHOLOGY"].values, df["EVIDENCES"].values):
        if c not in cidx:
            continue
        ci = cidx[c]; prior[ci] += 1; ctot[ci] += 1
        for e in set(base_code(e) for e in ast.literal_eval(evstr)):
            if e in eidx:
                count[ci, eidx[e]] += 1
    log_pe = np.log(count / ctot[:, None])              # [nC,nE]
    log_prior = np.log(prior / prior.sum())
    return log_prior, log_pe, cidx, eidx


def build_split(df, conditions, evid_codes, ask_vocab, diag_vocab, nb, out, n_max, seed):
    log_prior, log_pe, cidx, eidx = nb
    n_ask = len(ask_vocab)
    DIAG_TOK = len(evid_codes)          # special DIAG token id
    PAD_TOK = len(evid_codes) + 1
    rng = np.random.default_rng(seed)
    toks, extras, acts, mstates = [], [], [], []
    sub = df[df["PATHOLOGY"] == TARGET]
    for evstr, init in zip(sub["EVIDENCES"].values, sub["INITIAL_EVIDENCE"].values):
        evs = ast.literal_eval(evstr)
        # Asking order: a canonical (informativeness-priority) order lightly
        # perturbed by SHUFFLE_FRAC adjacent swaps.  This models a realistic,
        # largely-consistent diagnostic routine that still carries per-step
        # uncertainty (neither a fully-sorted nor a random sequence).
        rest = sorted({base_code(e) for e in evs if base_code(e) != base_code(init)},
                      key=lambda e: ask_vocab.get(e, 10**6))
        for i in range(len(rest) - 1):
            if rng.random() < SHUFFLE_FRAC:
                rest[i], rest[i + 1] = rest[i + 1], rest[i]
        order = [base_code(init)] + rest
        bases = [e for e in order if e in ask_vocab]
        if len(bases) < 3:
            continue
        # incremental NB posterior
        logpost = log_prior.copy()
        seq_tok, seq_extra, seq_act = [], [], []
        for e in bases:
            ei = eidx.get(e, None)
            if ei is not None:
                logpost = logpost + log_pe[:, ei]
            p = np.exp(logpost - logpost.max()); p = p / p.sum()
            topk = np.sort(p)[::-1][:TOPK]
            if len(topk) < TOPK:
                topk = np.pad(topk, (0, TOPK - len(topk)))
            ent = float(-(p * np.log(p + 1e-12)).sum())
            seq_tok.append(eidx.get(e, DIAG_TOK))
            seq_extra.append(np.concatenate([topk, [ent / np.log(len(conditions))]]))
            seq_act.append(ask_vocab[e])
        # final DIAG action
        seq_tok.append(DIAG_TOK)
        p = np.exp(logpost - logpost.max()); p = p / p.sum()
        topk = np.sort(p)[::-1][:TOPK]
        if len(topk) < TOPK:
            topk = np.pad(topk, (0, TOPK - len(topk)))
        seq_extra.append(np.concatenate([topk, [float(-(p * np.log(p + 1e-12)).sum()) / np.log(len(conditions))]]))
        seq_act.append(diag_vocab[TARGET])
        n = len(seq_act)
        # phase labels (1..5 -> 0..4)
        phase = [min(N_PHASE - 1, int(i / n * N_PHASE)) for i in range(n)]
        # left-pad to at least T
        if n < T:
            padn = T - n
            seq_tok = [PAD_TOK] * padn + seq_tok
            seq_extra = [np.zeros(TOPK + 1)] * padn + seq_extra
            seq_act = [ask_vocab.get(bases[0], 0)] * padn + seq_act
            phase = [0] * padn + phase
            n = T
        tok = np.array(seq_tok); extra = np.array(seq_extra, dtype=np.float32)
        act = np.array(seq_act); ph = np.array(phase)
        for s in range(0, n - T + 1):
            toks.append(tok[s:s + T]); extras.append(extra[s:s + T])
            acts.append(act[s:s + T]); mstates.append(ph[s:s + T])
        if len(acts) >= n_max:
            break
    tok = np.array(toks, dtype=np.int64)
    extra = np.array(extras, dtype=np.float32)
    act = np.array(acts, dtype=np.int64)
    mstate = np.array(mstates, dtype=np.int64)
    if len(act) > n_max:
        idx = rng.choice(len(act), n_max, replace=False)
        tok, extra, act, mstate = tok[idx], extra[idx], act[idx], mstate[idx]
    np.savez(out, tok=tok, extra=extra, act=act, mstate=mstate)
    return act, DIAG_TOK, PAD_TOK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data_raw/ddxplus")
    ap.add_argument("--out", default="../data_proc/ddxplus")
    ap.add_argument("--n_train", type=int, default=165000)
    ap.add_argument("--n_val", type=int, default=25000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    conditions = list(json.load(open(os.path.join(args.raw, "release_conditions.json"))).keys())
    evid = json.load(open(os.path.join(args.raw, "release_evidences.json")))
    evid_codes = sorted(evid.keys())

    print("loading train patients...")
    tr = pd.read_csv(os.path.join(args.raw, "ext/release_train_patients"))
    print(f"  {len(tr)} patients; URTI={int((tr['PATHOLOGY']==TARGET).sum())}")
    va = pd.read_csv(os.path.join(args.raw, "ext/release_validate_patients"))

    print("building Naive-Bayes teacher...")
    nb = build_nb(tr, conditions, evid_codes)

    # Action vocabulary = full agent action space (Appendix C.4, 225 actions):
    # ASK over ALL evidence codes (the agent may query any evidence) + 2 DIAG
    # actions.  URTI trajectories only exercise ~20 ASK + DIAG-URTI, leaving many
    # rare/unused actions -- the extreme imbalance the paper stresses.
    ask_vocab = {e: i for i, e in enumerate(evid_codes)}       # 223 ASK actions
    n_ask = len(ask_vocab)
    diag_vocab = {TARGET: n_ask}                               # DIAG-URTI = 223
    for c in conditions:
        diag_vocab.setdefault(c, n_ask + 1)                   # DIAG-other = 224
    n_actions = n_ask + 2
    print(f"  n_ask={n_ask} n_diag=2 n_actions={n_actions}")

    print("building train split...")
    build_split(tr, conditions, evid_codes, ask_vocab, diag_vocab, nb,
                os.path.join(args.out, "train.npz"), args.n_train, seed=1)
    print("building val split...")
    act, DIAG_TOK, PAD_TOK = build_split(va, conditions, evid_codes, ask_vocab, diag_vocab, nb,
                                         os.path.join(args.out, "val.npz"), args.n_val, seed=2)

    meta = dict(n_actions=n_actions, vocab_size=len(evid_codes) + 2, extra_dim=TOPK + 1,
                n_mental=N_PHASE, n_ask=n_ask, n_diag=len(conditions))
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=2)
    d = np.load(os.path.join(args.out, "train.npz"))
    print("train", d["act"].shape, "unique actions", len(np.unique(d["act"])))
    print("meta", meta)


if __name__ == "__main__":
    main()
