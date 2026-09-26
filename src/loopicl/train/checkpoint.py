"""Checkpoint save/load/management utilities."""

from __future__ import annotations

import os
import torch


def get_latest_checkpoint(ckpt_dir: str) -> str | None:
    """Return the path to the latest step checkpoint in `ckpt_dir`, or None."""
    if not os.path.isdir(ckpt_dir):
        return None

    checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
    if not checkpoints:
        return None

    try:
        latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))[-1]
        return os.path.join(ckpt_dir, latest)
    except Exception as e:
        print(f"Error parsing checkpoint filenames: {e}")
        return None


def save_checkpoint(ckpt_dir: str, name: str, model_config, raw_model, optimizer, scheduler, curr_step: int, ema_model=None):
    """Save model and training state to `ckpt_dir/name`."""
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint = {
        "config": model_config,
        "state_dict": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "curr_step": curr_step,
    }
    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()
    torch.save(checkpoint, os.path.join(ckpt_dir, name))


def manage_checkpoints(ckpt_dir: str, limit: int, save_perm_every: int):
    """Delete oldest temporary checkpoints when count exceeds `limit`."""
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
    temp_checkpoints = []
    for ckpt in checkpoints:
        try:
            step = int(ckpt.split("-")[1].split(".")[0])
            if step % save_perm_every != 0:
                temp_checkpoints.append((step, ckpt))
        except Exception:
            continue

    temp_checkpoints.sort(key=lambda x: x[0])

    for _, ckpt_name in temp_checkpoints[: max(0, len(temp_checkpoints) - limit)]:
        try:
            os.remove(os.path.join(ckpt_dir, ckpt_name))
        except Exception as e:
            print(f"Error removing checkpoint {ckpt_name}: {e}")
