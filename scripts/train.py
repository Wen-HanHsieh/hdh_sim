"""Train an HDM (default) or HDH (--history) verifier on one task's rollouts.

Every query of an episode is labeled with that episode's terminal outcome:
    loss = BCEWithLogits(f(x, U), 1 - success),   Q(x, U) = 1 - sigmoid(f)

    python scripts/train.py --data data/libero_10_0 --norm_stats $CKPT/assets/hdh/libero_10/norm_stats.json \
        --out checkpoints/libero_10_0_hdh_s0.pt --history --seed 0
"""
import argparse
import glob
import json

import numpy as np
import torch
import torch.nn.functional as F

from hdh.model import HDH, pool, save

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--norm_stats", required=True, help="norm_stats.json of the pi0.5 checkpoint")
ap.add_argument("--out", required=True)
ap.add_argument("--history", action="store_true")
ap.add_argument("--hmax", type=int, default=3)
ap.add_argument("--horizon", type=int, default=8)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

stats = json.load(open(a.norm_stats))["norm_stats"]
norm = {"action_mean": np.float32(stats["actions"]["mean"][:7]), "action_std": np.float32(stats["actions"]["std"][:7]),
        "state_mean": np.float32(stats["state"]["mean"][:8]), "state_std": np.float32(stats["state"]["std"][:8])}
H, dev = a.horizon, "cuda" if torch.cuda.is_available() else "cpu"


def load_split(split):
    """Queries whose executed length is below H are dropped (never-executed actions)."""
    eps = []
    for path in sorted(glob.glob(f"{a.data}/ep_*.npz")):
        z = np.load(path)
        keep = np.flatnonzero(z["executed"] >= H)
        if str(z["split"]) != split or not len(keep):
            continue
        eps.append({"tok": z["tok"][keep],
                    "pro": ((z["proprio"][keep] - norm["state_mean"]) / norm["state_std"]).astype(np.float32),
                    "U": ((z["chunk"][keep, :H, :7] - norm["action_mean"]) / norm["action_std"]).astype(np.float32),
                    "fail": 1.0 - int(z["success"])})
    return eps


def batch(eps, queries):
    """Context, prefixes and labels for (episode, query) pairs; history = preceding queries."""
    tok = np.stack([eps[e]["tok"][t] for e, t in queries]).astype(np.float32)
    pro = np.stack([eps[e]["pro"][t] for e, t in queries])
    U = np.stack([eps[e]["U"][t] for e, t in queries])
    y = np.array([eps[e]["fail"] for e, _ in queries], np.float32)
    hist = None
    if a.history:
        n, W = len(queries), tok.shape[-1]
        vis, hp = np.zeros((n, a.hmax, W), np.float32), np.zeros((n, a.hmax, 8), np.float32)
        act, mask = np.zeros((n, a.hmax, H, 7), np.float32), np.zeros((n, a.hmax), np.float32)
        for b, (e, t) in enumerate(queries):
            for s, j in enumerate(range(max(0, t - a.hmax), t)):
                vis[b, s], hp[b, s], act[b, s], mask[b, s] = pool(eps[e]["tok"][j]), eps[e]["pro"][j], eps[e]["U"][j], 1
        hist = tuple(torch.from_numpy(x).to(dev) for x in (vis, hp, act, mask))
    tok, pro, U, y = (torch.from_numpy(x).to(dev) for x in (tok, pro, U, y))
    return model.context(tok, pro, hist), U, y


@torch.no_grad()
def auc(eps):
    """Query-level ROC AUC of P(failure) on held-out episodes."""
    model.eval()
    p, y = [], []
    for e, ep in enumerate(eps):
        ctx, U, label = batch(eps, [(e, t) for t in range(len(ep["U"]))])
        p.append(torch.sigmoid(model(ctx, U)).cpu().numpy())
        y.append(label.cpu().numpy())
    p, y = np.concatenate(p), np.concatenate(y)
    pos, neg = p[y == 1], p[y == 0]
    if not (len(pos) and len(neg)):
        return float("nan")
    return float((pos[:, None] > neg).mean() + 0.5 * (pos[:, None] == neg).mean())


train, val = load_split("train"), load_split("val")
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
config = dict(token_dim=train[0]["tok"].shape[-1], horizon=H, action_dim=7, history=a.history, hmax=a.hmax)
model = HDH(**config).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
steps_per_epoch = max(1, sum(len(e["U"]) for e in train) // a.batch)
for _ in range(a.epochs):
    model.train()
    for _ in range(steps_per_epoch):
        # episode-uniform, then query-uniform: long episodes must not dominate the label
        ep = rng.integers(0, len(train), a.batch)
        ctx, U, y = batch(train, [(e, rng.integers(0, len(train[e]["U"]))) for e in ep])
        loss = F.binary_cross_entropy_with_logits(model(ctx, U), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

save(a.out, model, config, norm)
print(f"val AUC {auc(val):.4f} -> {a.out}")
