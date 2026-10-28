"""Register the pi0.5 LIBERO fine-tuning config in an openpi checkout (idempotent).

    python scripts/add_config.py --openpi /path/to/openpi
"""
import argparse
import pathlib
import re

NAME = "pi05_libero_hdh"
CONFIG = f'''    TrainConfig(
        name="{NAME}",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(repo_id="hdh/libero_10",
            base_config=DataConfig(prompt_from_task=True), extra_delta_transform=False),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=5_000,
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=5_000, decay_lr=2.5e-6),
        freeze_filter=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False,
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora").get_freeze_filter(),
        ema_decay=None,
    ),
'''

ap = argparse.ArgumentParser()
ap.add_argument("--openpi", required=True, type=pathlib.Path)
path = ap.parse_args().openpi / "src" / "openpi" / "training" / "config.py"
src = path.read_text()
if f'name="{NAME}"' not in src:
    anchor = re.search(r"^_CONFIGS(: list\[TrainConfig\])? = \[\n", src, re.M)
    assert anchor, f"_CONFIGS list not found in {path}"
    path.write_text(src[:anchor.end()] + CONFIG + src[anchor.end():])
print(f"{NAME} registered in {path}")
