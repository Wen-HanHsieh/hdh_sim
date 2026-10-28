"""Build the fine-tuning dataset: 10 public LIBERO demonstrations for each of five LIBERO-10
tasks, pinned by episode index and checked against per-episode checksums.
Run in the openpi environment.

    python scripts/build_dataset.py
"""
import hashlib
import io
import json
import os
import shutil

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from PIL import Image

REPO_ID = "hdh/libero_10"
manifest = json.load(open(os.path.join(os.path.dirname(__file__), "..", "benchmark", "finetune_episodes.json")))
shutil.rmtree(HF_LEROBOT_HOME / REPO_ID, ignore_errors=True)
image = {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]}
dataset = LeRobotDataset.create(
    repo_id=REPO_ID, robot_type="panda", fps=10, image_writer_threads=10, image_writer_processes=5,
    features={"image": image, "wrist_image": image,
              "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
              "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]}})


def decode(cell):
    return np.array(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


for task in manifest["tasks"]:
    for episode, checksum in task["episodes"].items():
        i = int(episode)
        table = pq.read_table(hf_hub_download(manifest["source"], f"data/chunk-{i // 1000:03d}/episode_{i:06d}.parquet",
                                              repo_type="dataset"))
        state = np.asarray(table.column("state").to_pylist(), np.float32)
        actions = np.asarray(table.column("actions").to_pylist(), np.float32)
        assert hashlib.sha256(actions.tobytes() + state.tobytes()).hexdigest()[:16] == checksum, f"episode {i} changed"
        for img, wrist, s, act in zip(table.column("image").to_pylist(), table.column("wrist_image").to_pylist(), state, actions):
            dataset.add_frame({"image": decode(img), "wrist_image": decode(wrist), "state": s, "actions": act,
                               "task": task["instruction"]})
        dataset.save_episode()
    print(f"{task['task']}: {len(task['episodes'])} episodes", flush=True)
