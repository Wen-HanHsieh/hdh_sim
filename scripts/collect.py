"""Roll out the frozen policy (K=1) on 100 training states and save, per episode, each
query's prefix tokens, proprioception, sampled chunk and the terminal success label.

    python scripts/collect.py --task_id 0 --sock /tmp/pi.sock --out data
"""
import argparse
import os

import numpy as np

from hdh.client import RemotePolicy
from hdh.env import LiberoTask
from hdh.utils import proprio_state

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_10")
ap.add_argument("--task_id", type=int, required=True)
ap.add_argument("--sock", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--horizon", type=int, default=8)
ap.add_argument("--val_frac", type=float, default=0.2)
a = ap.parse_args()

env = LiberoTask(a.suite, a.task_id)
policy = RemotePolicy(a.sock)
out = f"{a.out}/{a.suite}_{a.task_id}"
os.makedirs(out, exist_ok=True)
states, seeds = env.training_states(a.episodes)
n_val = int(a.episodes * a.val_frac)

for i, (state, seed) in enumerate(zip(states, seeds)):
    path = f"{out}/ep_{i:04d}.npz"
    if os.path.exists(path):
        continue
    obs = env.reset(state, seed)
    policy.begin(env.name, seed, "collect")
    tok, pro, chunk, executed = [], [], [], []
    steps, done = 0, False
    while not done and steps < env.max_steps:
        tok.append(policy.tokens(obs, env.language))
        pro.append(proprio_state(obs))
        chunk.append(policy.sample(obs, env.language, 1)[0])
        n = 0
        for action in chunk[-1][:a.horizon]:
            obs, _, done, _ = env.step(action)
            steps, n = steps + 1, n + 1
            if done or steps >= env.max_steps:
                break
        executed.append(n)
    np.savez_compressed(path, tok=np.asarray(tok, np.float16), proprio=np.asarray(pro, np.float32),
                        chunk=np.asarray(chunk, np.float32), executed=np.asarray(executed, np.int32),
                        success=int(done), split="val" if i < n_val else "train")
    print(f"{env.name} ep {i:3d} success={int(done)} queries={len(executed)}", flush=True)
