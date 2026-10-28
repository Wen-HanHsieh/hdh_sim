from multiprocessing.connection import Client

import numpy as np

from hdh.utils import OBS_KEYS


class RemotePolicy:
    """Torch-side handle to a running `hdh.server`. All arms of a paired comparison must
    share one server: pi0.5 samples differ slightly across server restarts."""

    def __init__(self, sock):
        self.conn = Client(sock, "AF_UNIX")

    def _call(self, op, **kw):
        self.conn.send({"op": op, **kw})
        r = self.conn.recv()
        if "error" in r:
            raise RuntimeError(r["error"])
        return r.get("out")

    def begin(self, task, seed, salt=""):
        self._call("begin", task=task, seed=int(seed), salt=salt)

    def sample(self, obs, prompt, k):
        return np.asarray(self._call("sample", obs=_slim(obs), prompt=prompt, k=int(k)))

    def tokens(self, obs, prompt):
        return np.asarray(self._call("tokens", obs=_slim(obs), prompt=prompt), np.float32)

    def stop(self):
        self._call("stop")


def _slim(obs):
    return {k: np.asarray(obs[k]) for k in OBS_KEYS}
