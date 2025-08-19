# /amft-project/src/controller/adaptive_weight_controller.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List, Optional
import logging
from src.model.policy_model import PolicyModel
logger = logging.getLogger(__name__)


class AdaptiveWeightController:

    def __init__(
            self,
            initial_mu: float = 0.5,
            meta_lr: float = 1e-4,
            entropy_lr: float = 5e-4,
            clip_range: Tuple[float, float] = (0.05, 0.95),
    ):

        self.mu = torch.tensor(initial_mu, dtype=torch.float32)
        self.meta_lr = meta_lr
        self.entropy_lr = entropy_lr
        self.mu_min, self.mu_max = clip_range
        self.target_entropy: Optional[float] = None

        logger.info(f"AdaptiveWeightController initialized with mu={self.mu.item():.2f}, "
                    f"meta_lr={meta_lr}, entropy_lr={entropy_lr}")

    def set_target_entropy(self, target_entropy: float):
        self.target_entropy = target_entropy

    def get_mu(self) -> float:
        return self.mu.item()

    def update_mu(self, current_entropy: float, meta_gradient: float = 0.0):
        if self.target_entropy is None:
            raise ValueError("Target entropy has not been set. Call set_target_entropy() first.")

        entropy_heuristic_grad = self.target_entropy - current_entropy

        mu_update = self.meta_lr * meta_gradient + self.entropy_lr * entropy_heuristic_grad
        with torch.no_grad():
            self.mu += mu_update
            self.mu.clamp_(self.mu_min, self.mu_max)

    @torch.no_grad()
    def compute_meta_gradient(
            self,
            policy_model: PolicyModel,
            val_dataloader: DataLoader,
            sft_loss_grad: Tuple[torch.Tensor, ...],
            rl_loss_grad: Tuple[torch.Tensor, ...],
            policy_lr: float
    ) -> float:

        try:
            val_batch = next(iter(val_dataloader))
        except StopIteration:
            logger.warning("Validation dataloader exhausted. Meta-gradient will be zero.")
            return 0.0

        device = self.mu.device
        val_batch = {k: v.to(device) for k, v in val_batch.items()}


        grad_diff = [sft_g - rl_g for sft_g, rl_g in zip(sft_loss_grad, rl_loss_grad)]


        with torch.enable_grad():
            val_outputs = policy_model(**val_batch)
            val_logits = val_outputs['logits']

            val_utility_loss = F.cross_entropy(
                val_logits.view(-1, val_logits.size(-1)),
                val_batch['labels'].view(-1),
                ignore_index=policy_model.tokenizer.pad_token_id
            )
            outer_gradient = torch.autograd.grad(val_utility_loss, policy_model.parameters())

        dot_product = sum(torch.sum(og * gd) for og, gd in zip(outer_gradient, grad_diff))
        meta_grad = -policy_lr * dot_product

        return meta_grad.item()