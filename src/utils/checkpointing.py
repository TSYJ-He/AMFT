# /amft-project/src/utils/checkpointing.py

import torch
import os
import logging
from typing import Optional, Dict, Any
from src.model.policy_model import PolicyModel
logger = logging.getLogger(__name__)


def save_checkpoint(
        model: PolicyModel,
        optimizer_dict: Dict[str, torch.optim.Optimizer],
        step: int,
        output_dir: str,
        config: Optional[Dict[str, Any]] = None
):

    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Saving checkpoint for step {step} to {checkpoint_dir}...")
    model.get_base_model().save_pretrained(checkpoint_dir)
    model.tokenizer.save_pretrained(checkpoint_dir)

    torch.save(model.value_head.state_dict(), os.path.join(checkpoint_dir, "value_head.pt"))

    for name, optimizer in optimizer_dict.items():
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, f"{name}_optimizer.pt"))
    if config:
        torch.save(config, os.path.join(checkpoint_dir, "training_args.bin"))

    logger.info(f"Checkpoint successfully saved to {checkpoint_dir}")


def load_checkpoint(
        model: PolicyModel,
        optimizer_dict: Dict[str, torch.optim.Optimizer],
        checkpoint_path: str,
        device: torch.device
) -> Optional[Dict[str, Any]]:

    if not os.path.isdir(checkpoint_path):
        logger.error(f"Checkpoint path not found: {checkpoint_path}")
        return None

    logger.info(f"Loading checkpoint from {checkpoint_path}...")

    value_head_path = os.path.join(checkpoint_path, "value_head.pt")
    if os.path.exists(value_head_path):
        model.value_head.load_state_dict(torch.load(value_head_path, map_location=device))
        logger.info("Loaded value head state.")
    else:
        logger.warning(f"Value head state not found at {value_head_path}. Skipping.")

    for name, optimizer in optimizer_dict.items():
        optimizer_path = os.path.join(checkpoint_path, f"{name}_optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            logger.info(f"Loaded {name} optimizer state.")
        else:
            logger.warning(f"{name} optimizer state not found at {optimizer_path}. Skipping.")

    config_path = os.path.join(checkpoint_path, "training_args.bin")
    config = None
    if os.path.exists(config_path):
        config = torch.load(config_path)
        logger.info("Loaded training configuration.")

    logger.info(f"Checkpoint loading from {checkpoint_path} complete.")
    return config