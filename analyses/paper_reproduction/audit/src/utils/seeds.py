"""Deterministic random seed derivation protocol for reproducible audit execution."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any
import yaml
import numpy as np

MASTER_SEED_DEFAULT = 20260919


def derive_seed(master_seed: int, task_id: str) -> int:
    """Deterministically derive a 31-bit non-negative integer seed from master_seed and task_id.
    
    Formula: int(SHA256(f"{master_seed}:{task_id}")[:8], 16) % (2**31 - 1)
    """
    key = f"{master_seed}:{task_id}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def spawn_seeds(base_seed: int, n: int) -> list[int]:
    """Spawn N independent child seeds from a base task seed using numpy SeedSequence."""
    ss = np.random.SeedSequence(base_seed)
    children = ss.spawn(n)
    return [int(child.generate_state(1)[0] % (2**31 - 1)) for child in children]


def load_seeds_config(config_path: str | Path) -> dict[str, Any]:
    """Load and parse seeds.yaml configuration."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Seeds configuration not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def get_task_seed(task_id: str, config_path: str | Path | None = None) -> int:
    """Retrieve task seed from config if present, or derive dynamically."""
    if config_path is not None:
        cfg = load_seeds_config(config_path)
        master = cfg.get("master_seed", MASTER_SEED_DEFAULT)
        seeds_map = cfg.get("task_seeds", {})
        if task_id in seeds_map:
            return int(seeds_map[task_id])
        return derive_seed(master, task_id)
    return derive_seed(MASTER_SEED_DEFAULT, task_id)


def verify_seeds_config(config_path: str | Path) -> bool:
    """Verify that all task seeds in config_path match the deterministic derivation formula."""
    cfg = load_seeds_config(config_path)
    master = cfg.get("master_seed", MASTER_SEED_DEFAULT)
    task_seeds = cfg.get("task_seeds", {})

    if not task_seeds:
        raise ValueError(f"No task_seeds defined in {config_path}")

    mismatches: list[str] = []
    for task_id, stored_seed in task_seeds.items():
        expected_seed = derive_seed(master, task_id)
        if stored_seed != expected_seed:
            mismatches.append(
                f"Task {task_id}: stored {stored_seed} != derived {expected_seed}"
            )

    if mismatches:
        raise ValueError(
            f"Seed verification FAILED for {config_path}:\n" + "\n".join(mismatches)
        )

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic seed verification utility")
    parser.add_argument("--verify", action="store_true", help="Verify seed derivation against config")
    parser.add_argument("--config", type=str, default="env/seeds.yaml", help="Path to seeds.yaml")
    parser.add_argument("--task", type=str, default=None, help="Get seed for specific task")
    args = parser.parse_args()

    if args.verify:
        verify_seeds_config(args.config)
        print(f"✅ Seeds verified successfully against derivation formula ({args.config})")
    elif args.task:
        seed = get_task_seed(args.task, args.config)
        print(f"Seed for {args.task}: {seed}")
    else:
        parser.print_help()
