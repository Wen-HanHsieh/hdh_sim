"""Success rate per task (mean +- std over runs), macro average, and time per query.

    python scripts/analyze.py --runs runs
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="runs")
a = ap.parse_args()

success = collections.defaultdict(list)  # (arm, task, run) -> outcomes
ms_per_query = collections.defaultdict(list)  # arm -> per-episode ms/query
for path in glob.glob(f"{a.runs}/run*/*/*/*.json"):
    run, task, arm = path.split(os.sep)[-4:-1]
    r = json.load(open(path))
    success[arm, task, run].append(r["success"])
    ms_per_query[arm].append(1e3 * r["seconds"] / r["queries"])

arms = [x for x in ("baseline", "random", "hdm", "hdh") if x in ms_per_query]
tasks = sorted({t for _, t, _ in success})
print(f"{'':10s}" + "".join(f"{t:>16s}" for t in tasks) + f"{'macro':>8s}{'p50 ms':>9s}{'p95 ms':>9s}")
for arm in arms:
    means, row = [], ""
    for t in tasks:
        rates = [100 * np.mean(v) for (m, tt, _), v in sorted(success.items()) if m == arm and tt == t]
        means.append(np.mean(rates))
        row += f"{np.mean(rates):>10.1f} ±{np.std(rates):>4.1f}"
    p50, p95 = np.percentile(ms_per_query[arm], [50, 95])
    print(f"{arm:10s}{row}{np.mean(means):>8.1f}{p50:>9.0f}{p95:>9.0f}")
