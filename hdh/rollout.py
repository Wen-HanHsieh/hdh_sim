import time

import numpy as np
import torch

from hdh.model import load, pool
from hdh.utils import proprio_state


class Baseline:
    """Frozen policy: one sampled chunk per query."""
    k = 1

    def reset(self):
        pass

    def select(self, policy, obs, prompt, chunks):
        return 0


class Random(Baseline):
    """Uniform choice among the same K candidates the verifier sees."""

    def __init__(self, k, seed=0):
        self.k, self.rng = k, np.random.default_rng(seed)

    def select(self, policy, obs, prompt, chunks):
        return int(self.rng.integers(len(chunks)))


class Verifier(Baseline):
    """Executes the candidate with the highest predicted task-success score. Inputs are built
    exactly as in scripts/train.py: f16 tokens, normalized proprio and actions, and a history
    of the last `hmax` committed queries (empty at the first query)."""

    def __init__(self, path, k, device="cuda"):
        self.model, self.norm = load(path, device)
        self.k, self.device = k, device
        self.horizon = self.model.apos.shape[0]
        self.reset()

    def reset(self):
        self.hist, self.pending = [], None

    def _tensor(self, x):
        return torch.as_tensor(np.asarray(x, np.float32)[None], device=self.device)

    def _history(self, width):
        n = self.model.hmax
        vis, pro = np.zeros((n, width), np.float32), np.zeros((n, 8), np.float32)
        act, mask = np.zeros((n, self.horizon, 7), np.float32), np.zeros(n, np.float32)
        for i, (v, p, a) in enumerate(self.hist):
            vis[i], pro[i], act[i], mask[i] = v, p, a, 1.0
        return tuple(self._tensor(x) for x in (vis, pro, act, mask))

    def _actions(self, chunks):
        U = np.asarray(chunks, np.float32)[:, :self.horizon, :7]
        return ((U - self.norm["action_mean"]) / self.norm["action_std"]).astype(np.float32)

    @torch.no_grad()
    def score(self, policy, obs, prompt, chunks):
        """P(failure) for each candidate's executed prefix."""
        nm = self.norm
        tok = policy.tokens(obs, prompt).astype(np.float16)
        pro = ((proprio_state(obs) - nm["state_mean"]) / nm["state_std"]).astype(np.float32)
        hist = self._history(tok.shape[-1]) if self.model.history else None
        ctx = self.model.context(self._tensor(tok), self._tensor(pro), hist)
        U = self._actions(chunks)
        self.pending = (pool(tok), pro, U)
        logits = self.model(ctx.repeat(len(U), 1, 1), torch.as_tensor(U, device=self.device))
        return torch.sigmoid(logits).cpu().numpy()

    def select(self, policy, obs, prompt, chunks):
        k = int(np.argmin(self.score(policy, obs, prompt, chunks)))
        vis, pro, U = self.pending
        self.hist = (self.hist + [(vis, pro, U[k])])[-self.model.hmax:]
        return k


def run_episode(env, policy, selector, state, seed, horizon=8, salt=""):
    """One closed-loop episode: sample K chunks, select one, execute its first `horizon` actions."""
    obs = env.reset(state, seed)
    policy.begin(env.name, seed, salt)
    selector.reset()
    t0, steps, queries, done = time.time(), 0, 0, False
    while not done and steps < env.max_steps:
        chunks = policy.sample(obs, env.language, selector.k)
        executed = chunks[selector.select(policy, obs, env.language, chunks)][:horizon]
        queries += 1
        for a in executed:
            obs, _, done, _ = env.step(a)
            steps += 1
            if done or steps >= env.max_steps:
                break
    return {"task": env.name, "seed": int(seed), "success": bool(done), "steps": steps,
            "queries": queries, "seconds": time.time() - t0}
