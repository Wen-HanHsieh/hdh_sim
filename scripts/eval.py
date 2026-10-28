"""Closed-loop evaluation on the 50 held-out states of one task. All arms of a run share
the server and the sampling key, so they see identical candidates on identical states.

    python scripts/eval.py --task_id 0 --sock /tmp/pi.sock --arm hdh --ckpt checkpoints/libero_10_0_hdh_s0.pt --run 0
"""
import argparse
import json
import os

import numpy as np

from hdh.client import RemotePolicy
from hdh.env import LiberoTask
from hdh.rollout import Baseline, Random, Verifier, run_episode

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_10")
ap.add_argument("--task_id", type=int, required=True)
ap.add_argument("--sock", required=True)
ap.add_argument("--arm", required=True, choices=["baseline", "random", "hdm", "hdh"])
ap.add_argument("--ckpt", help="verifier checkpoint for --arm hdm/hdh")
ap.add_argument("--k", type=int, default=2)
ap.add_argument("--horizon", type=int, default=8)
ap.add_argument("--run", type=int, default=0)
ap.add_argument("--out", default="runs")
a = ap.parse_args()

if a.arm == "baseline":
    selector = Baseline()
elif a.arm == "random":
    selector = Random(a.k, seed=a.run)
else:
    selector = Verifier(a.ckpt, a.k)
    assert selector.model.history == (a.arm == "hdh"), f"{a.ckpt} is not an {a.arm.upper()} checkpoint"

env = LiberoTask(a.suite, a.task_id)
policy = RemotePolicy(a.sock)
states = np.load(os.path.join(os.path.dirname(__file__), "..", "benchmark", "eval_states", f"{a.suite}_{a.task_id}.npz"))
out = f"{a.out}/run{a.run}/{a.suite}_{a.task_id}/{a.arm}"
os.makedirs(out, exist_ok=True)
for state, seed in zip(states["states"], states["seeds"]):
    path = f"{out}/{seed}.json"
    if os.path.exists(path):
        continue
    r = run_episode(env, policy, selector, state, seed, a.horizon, salt=f"run{a.run}")
    json.dump(r, open(path, "w"))
    print(f"{env.name} {a.arm} seed={seed} success={int(r['success'])}", flush=True)
