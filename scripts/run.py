"""Main training entry point.

Usage:  python scripts/run.py --dataset car --seed 0
"""
import os, sys, json, argparse, random, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rgai.data import ArrayDataset, FrameDataset
from rgai.trainer import Trainer
from scripts.configs import CONFIGS, ROOT


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_datasets(cfg):
    dd = cfg.data_dir
    if cfg.model.backbone == "cnn":
        tr = FrameDataset(os.path.join(dd, "train_meta.npz"), cfg.model.n_actions, cfg.model.img_size)
        va = FrameDataset(os.path.join(dd, "val_meta.npz"), cfg.model.n_actions, cfg.model.img_size)
    else:
        tr = ArrayDataset(os.path.join(dd, "train.npz"), cfg.model.n_actions)
        va = ArrayDataset(os.path.join(dd, "val.npz"), cfg.model.n_actions)
        # infer evidence2vec dims from data meta if present
        meta = os.path.join(dd, "meta.json")
        if os.path.exists(meta):
            m = json.load(open(meta))
            cfg.model.vocab_size = m.get("vocab_size", cfg.model.vocab_size)
            cfg.model.extra_dim = m.get("extra_dim", cfg.model.extra_dim)
            cfg.model.n_actions = m.get("n_actions", cfg.model.n_actions)
            cfg.model.n_mental = m.get("n_mental", cfg.model.n_mental)
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CONFIGS.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--game", default="berzerk")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n_mental", type=int, default=None, help="override mental-state cardinality K (Fig 12-14 sweep)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.dataset == "atari":
        cfg = CONFIGS[args.dataset](args.seed, game=args.game)
    else:
        cfg = CONFIGS[args.dataset](args.seed)
    if args.epochs:
        cfg.train.full_epochs = args.epochs
    if args.n_mental:
        cfg.model.n_mental = args.n_mental
    set_seed(cfg.train.seed)

    out = os.path.join(ROOT, "runs", f"{cfg.name}_s{args.seed}{('_'+args.tag) if args.tag else ''}")
    os.makedirs(out, exist_ok=True)
    cfg.out_dir = out
    logf = open(os.path.join(out, "log.txt"), "w")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    log(f"===== {cfg.name} seed={args.seed} device={cfg.train.device} =====")
    tr, va = load_datasets(cfg)
    log(f"train={len(tr)} val={len(va)} actions={cfg.model.n_actions} mental={cfg.model.n_mental}")
    cfg.to_json(os.path.join(out, "config.json"))

    trainer = Trainer(cfg, tr, va, log=log)
    res = trainer.fit()

    # final eval with rules
    from rgai.metrics import evaluate
    final = evaluate(trainer.model, trainer.rules if len(trainer.rules) else None,
                     trainer.val_loader, cfg.train, cfg.train.device)
    res["final"] = final
    res["n_rules"] = len(trainer.rules)
    log("FINAL:", json.dumps(final, indent=2))
    log(f"convergence_time={res['convergence_time_h']:.3f}h peak_mem={res['peak_mem_mb']:.0f}MB rules={len(trainer.rules)}")
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=2)
    torch.save({"model": trainer.model.state_dict(), "rules": trainer.rules.state_dict()},
               os.path.join(out, "ckpt.pt"))


if __name__ == "__main__":
    main()
