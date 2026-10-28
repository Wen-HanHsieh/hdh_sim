import hashlib
import os

import numpy as np
import torch

# LIBERO loads its init states with torch.load; torch>=2.6 needs these allow-listed.
torch.serialization.add_safe_globals(
    [np.core.multiarray._reconstruct, np.ndarray, np.dtype]
    + ([np.dtypes.Float64DType] if hasattr(np, "dtypes") else []))

MAX_STEPS = {"libero_10": 520, "libero_goal": 300, "libero_spatial": 300, "libero_object": 300}
SETTLE_STEPS = 10
IDLE = [0.0] * 6 + [-1.0]


class LiberoTask:
    def __init__(self, suite, task_id, resolution=256):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        bench = benchmark.get_benchmark_dict()[suite]()
        task = bench.get_task(task_id)
        self.name = f"{suite}:{task_id}"
        self.language = task.language
        self.max_steps = MAX_STEPS[suite]
        self.init_states = bench.get_task_init_states(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution,
                                      camera_widths=resolution)

    def reset(self, state, seed):
        # LIBERO re-randomizes fixture poses from the global RNG on every reset; seeding it
        # makes two resets of the same state identical.
        np.random.seed(int(seed) % (2 ** 31 - 1))
        self.env.reset()
        obs = self.env.set_init_state(state)
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = self.env.step(IDLE)
        return obs

    def step(self, action):
        return self.env.step(np.asarray(action).tolist())

    def training_states(self, n=100):
        """Seeded resets, disjoint from LIBERO's canonical init states and from the
        held-out evaluation states (which use base 8e6)."""
        offset = int(hashlib.sha1(self.name.encode()).hexdigest()[:6], 16)
        canon = np.asarray(self.init_states)[:50]
        states, seeds = [], []
        for seed in range(6_000_000 + offset, 6_000_000 + offset + 3 * n):
            np.random.seed(seed)
            self.env.seed(seed)
            self.env.reset()
            s = np.asarray(self.env.env.sim.get_state().flatten())
            k = min(len(s), canon.shape[1])
            if not any(np.allclose(s[:k], c[:k], atol=1e-9) for c in canon):
                states.append(s)
                seeds.append(seed)
            if len(states) == n:
                return np.stack(states), np.array(seeds)
        raise RuntimeError(f"only {len(states)} disjoint training states for {self.name}")
