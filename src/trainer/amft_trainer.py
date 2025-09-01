# /amft-project/src/trainer/amft_trainer.py

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW
from typing import Dict, Any, Optional
import logging
import os
from src.model.policy_model import PolicyModel
from src.controller.adaptive_weight_controller import AdaptiveWeightController
from src.trainer.ppo_loss import compute_ppo_loss
from src.utils.logging import log_metrics
logger = logging.getLogger(__name__)
class AMFTTrainer:

    def __init__(
            self,
            config: Dict[str, Any],
            policy_model: PolicyModel,
            sft_dataloader: DataLoader,
            val_dataloader: DataLoader,
            # In a real implementation, environment would be passed here for rollouts
            # env: Any
    ):

        self.config = config
        self.policy_model = policy_model
        self.sft_dataloader = sft_dataloader
        self.val_dataloader = val_dataloader

        # Initialize optimizers
        self.policy_optimizer = AdamW(
            self.policy_model.get_base_model().parameters(),
            lr=config['policy_lr']
        )
        self.value_optimizer = AdamW(
            self.policy_model.value_head.parameters(),
            lr=config['value_lr']
        )

        # Initialize the core component: the adaptive weight controller
        self.adaptive_controller = AdaptiveWeightController(
            initial_mu=config['controller']['initial_mu'],
            meta_lr=config['controller']['meta_lr'],
            entropy_lr=config['controller']['entropy_lr'],
            clip_range=(config['controller']['mu_min'], config['controller']['mu_max'])
        )

        self.device = next(policy_model.parameters()).device
        logger.info(f"AMFT Trainer initialized. Model is on device: {self.device}")

    def train(self):
        self._sft_warmup()
        self._main_adaptive_loop()
        logger.info("AMFT Training finished.")

    def _sft_warmup(self):

        logger.info(f"Starting SFT warm-up for {self.config['warmup_steps']} steps...")
        self.policy_model.train()

        sft_iter = iter(self.sft_dataloader)

        for step in range(self.config['warmup_steps']):
            try:
                batch = next(sft_iter)
            except StopIteration:
                sft_iter = iter(self.sft_dataloader)
                batch = next(sft_iter)

            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.policy_model(**batch)
            logits = outputs['logits']

            # Standard SFT loss (cross-entropy)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1),
                ignore_index=self.policy_model.tokenizer.pad_token_id
            )

            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

            if (step + 1) % self.config['logging_steps'] == 0:
                logger.info(f"Warm-up Step [{step + 1}/{self.config['warmup_steps']}], SFT Loss: {loss.item():.4f}")

        target_entropy = self._calculate_target_entropy()
        self.adaptive_controller.set_target_entropy(target_entropy)
        logger.info(f"SFT warm-up finished. Target entropy set to: {target_entropy:.4f}")

    def _main_adaptive_loop(self):

        logger.info(f"Starting main adaptive training loop for {self.config['total_steps']} steps...")
        self.policy_model.train()

        sft_iter = iter(self.sft_dataloader)

        for step in range(self.config['total_steps']):

            try:
                sft_batch = next(sft_iter)
            except StopIteration:
                sft_iter = iter(self.sft_dataloader)
                sft_batch = next(sft_iter)
            sft_batch = {k: v.to(self.device) for k, v in sft_batch.items()}


            rl_prompts = {k: v for k, v in sft_batch.items() if k != 'labels'}
            rl_rollouts = self._perform_rollouts(rl_prompts)  # Placeholder function

            sft_outputs = self.policy_model(**sft_batch)
            sft_logits = sft_outputs['logits']
            sft_loss = F.cross_entropy(
                sft_logits.view(-1, sft_logits.size(-1)),
                sft_batch['labels'].view(-1),
                ignore_index=self.policy_model.tokenizer.pad_token_id
            )
            rl_loss, value_loss, metrics = compute_ppo_loss(
                self.policy_model,
                rl_rollouts  # contains rewards, advantages, etc.
            )
            current_entropy = metrics['policy_entropy'].mean()
            meta_grad = 0.0

            if step % self.config['controller']['meta_update_freq'] == 0:
                # Placeholder for meta-gradient calculation
                meta_grad = self.adaptive_controller.compute_meta_gradient(
                    policy_model=self.policy_model,
                    val_dataloader=self.val_dataloader,
                    sft_loss_grad=torch.autograd.grad(sft_loss, self.policy_model.parameters(), retain_graph=True),
                    rl_loss_grad=torch.autograd.grad(rl_loss, self.policy_model.parameters(), retain_graph=True),
                    policy_lr=self.config['policy_lr']
                )

            self.adaptive_controller.update_mu(
                current_entropy=current_entropy,
                meta_gradient=meta_grad
            )
            mu = self.adaptive_controller.get_mu()

            total_loss = (1 - mu) * rl_loss + mu * sft_loss

            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config['max_grad_norm'])
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

            if (step + 1) % self.config['logging_steps'] == 0:
                log_data = {
                    "step": step + 1,
                    "total_loss": total_loss.item(),
                    "sft_loss": sft_loss.item(),
                    "rl_loss": rl_loss.item(),
                    "value_loss": value_loss.item(),
                    "adaptive_weight_mu": mu,
                    "policy_entropy": current_entropy.item(),
                    **metrics
                }
                log_metrics(log_data)  # Placeholder for logging to console/WandB
                logger.info(
                    f"Step [{step + 1}/{self.config['total_steps']}], Mu: {mu:.4f}, Total Loss: {total_loss.item():.4f}")
            if (step + 1) % self.config['save_steps'] == 0:
                self._save_checkpoint(step + 1)

    @torch.no_grad()
    def _calculate_target_entropy(self) -> float:
        self.policy_model.eval()
        total_entropy = 0
        num_batches = 0
        for batch in self.sft_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.policy_model(**batch)
            logits = outputs['logits']

            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs, dim=-1)

            mask = (batch['input_ids'] != self.policy_model.tokenizer.pad_token_id).float()
            masked_entropy = (entropy * mask).sum() / mask.sum()

            total_entropy += masked_entropy.item()
            num_batches += 1
            if num_batches >= 20:
                break
        self.policy_model.train()
        return total_entropy / num_batches

    def _perform_rollouts(self, prompts: Dict[str, Any]) -> Dict[str, Any]:

        return {
            "prompts": prompts,
            "log_probs_old": torch.randn(prompts['input_ids'].shape[0], prompts['input_ids'].shape[1] - 1).to(
                self.device),
            "values_old": torch.randn(prompts['input_ids'].shape[0]).to(self.device),
            "actions": prompts['labels'][:, 1:],
            "advantages": torch.randn(prompts['input_ids'].shape[0], prompts['input_ids'].shape[1] - 1).to(self.device),
            "returns": torch.randn(prompts['input_ids'].shape[0], prompts['input_ids'].shape[1] - 1).to(self.device),
        }

    def _save_checkpoint(self, step: int):
        output_dir = os.path.join(self.config['output_dir'], f"checkpoint-{step}")
        os.makedirs(output_dir, exist_ok=True)
        self.policy_model.base_model.save_pretrained(output_dir)
        self.policy_model.tokenizer.save_pretrained(output_dir)
        torch.save(self.policy_model.value_head.state_dict(), os.path.join(output_dir, "value_head.pt"))
        logger.info(f"Checkpoint saved to {output_dir}")
