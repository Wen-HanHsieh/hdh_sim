"""Frozen pi0.5 policy host. Runs in the openpi (JAX) environment and serves samples and
prefix tokens over a unix socket to the torch side (env, verifier).

    python -m hdh.server /tmp/pi.sock pi05_libero_hdh /path/to/checkpoint
"""
import argparse
import traceback
from multiprocessing.connection import Listener

import numpy as np

from hdh.utils import proprio_state, stable_key


class Pi05:
    def __init__(self, config, checkpoint, resize=224):
        from openpi.policies import policy_config
        from openpi.training import config as _config

        self.policy = policy_config.create_trained_policy(_config.get_config(config), checkpoint)
        self.resize = resize
        self.key, self.query, self._tap = 0, 0, None

    def begin(self, task, seed, salt=""):
        self.key, self.query = stable_key(task, seed, salt), 0

    def _element(self, obs, prompt):
        from openpi_client import image_tools

        def image(x):  # 180-degree rotation, as in openpi's LIBERO preprocessing
            x = np.ascontiguousarray(np.asarray(x)[::-1, ::-1])
            return image_tools.convert_to_uint8(image_tools.resize_with_pad(x, self.resize, self.resize))

        return {"observation/image": image(obs["agentview_image"]),
                "observation/wrist_image": image(obs["robot0_eye_in_hand_image"]),
                "observation/state": proprio_state(obs), "prompt": prompt}

    def sample(self, obs, prompt, k):
        """K action chunks from independent flow-matching noise. Candidate 0 uses the key
        the K=1 baseline draws, so every arm sees the same first candidate."""
        import jax

        el = self._element(obs, prompt)
        chunks = []
        for c in range(k):
            parts = (self.key, self.query) if c == 0 else (self.key, self.query, c)
            self.policy._rng = jax.random.key(stable_key(*parts))
            chunks.append(self.policy.infer(el)["actions"])
        self.query += 1
        return np.asarray(chunks, np.float64)

    def tokens(self, obs, prompt):
        """SigLIP prefix tokens of the two camera views, 16x16 -> 8x8 by 2x2 mean pooling.
        Does not touch the sampler key."""
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model

        if self._tap is None:
            def tap(o):
                tok = self.policy._model.embed_prefix(o)[0][0, :512].reshape(2, 16, 16, -1)
                return tok.reshape(2, 8, 2, 8, 2, -1).mean((2, 4)).astype(jnp.bfloat16)
            self._tap = jax.jit(tap)
        inputs = self.policy._input_transform(dict(self._element(obs, prompt)))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        o = _model.preprocess_observation(None, _model.Observation.from_dict(inputs), train=False)
        return np.asarray(self._tap(o), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sock")
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    a = ap.parse_args()

    policy = Pi05(a.config, a.checkpoint)
    ops = {"begin": policy.begin, "sample": policy.sample, "tokens": policy.tokens}
    with Listener(a.sock, "AF_UNIX") as listener:
        print(f"serving on {a.sock}", flush=True)
        while True:
            with listener.accept() as conn:
                while True:
                    try:
                        req = conn.recv()
                    except EOFError:
                        break
                    op = req.pop("op")
                    if op == "stop":
                        conn.send({})
                        return
                    try:
                        conn.send({"out": ops[op](**req)})
                    except Exception:
                        conn.send({"error": traceback.format_exc()})


if __name__ == "__main__":
    main()
