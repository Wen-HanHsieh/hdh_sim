import numpy as np

OBS_KEYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat",
            "robot0_gripper_qpos")


def stable_key(*parts):
    """FNV-1a hash; unlike hash(), identical across processes."""
    h = 0x811C9DC5
    for ch in "|".join(map(str, parts)).encode():
        h = ((h ^ ch) * 0x01000193) & 0xFFFFFFFF
    return h


def quat2axisangle(q):
    q = np.asarray(q, np.float64)
    if q[3] > 1.0:
        q = q / np.linalg.norm(q)
    den = np.sqrt(max(1.0 - q[3] ** 2, 0.0))
    if den < 1e-8:
        return np.zeros(3)
    return q[:3] * 2.0 * np.arccos(np.clip(q[3], -1.0, 1.0)) / den


def proprio_state(obs):
    """8-D state used by pi0.5-LIBERO: eef position, axis-angle, gripper qpos."""
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]])
