# /amft-project/src/trainer/ppo_loss.py

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from src.model.policy_model import PolicyModel


def compute_ppo_loss(
        policy_model: PolicyModel,
        rollouts: Dict[str, torch.Tensor],
        ppo_clip_epsilon: float = 0.2,
        value_loss_coeff: float = 0.1,
        entropy_coeff: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:

    prompts = rollouts['prompts']
    actions = rollouts['actions']
    log_probs_old = rollouts['log_probs_old']
    values_old = rollouts['values_old']
    advantages = rollouts['advantages']
    returns = rollouts['returns']


    outputs = policy_model(**prompts)
    logits, values_new = outputs['logits'], outputs['value']

    action_log_prob_data = policy_model.get_action_log_prob_and_entropy(logits, actions)
    log_probs_new = action_log_prob_data['log_prob']
    entropy = action_log_prob_data['entropy']

    attention_mask = prompts.get("attention_mask")[:, 1:]  # Align with actions
    ratio = torch.exp(log_probs_new - log_probs_old)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    unclipped_loss = ratio * advantages
    clipped_loss = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon) * advantages
    policy_loss = -torch.min(unclipped_loss, clipped_loss)

    masked_policy_loss = (policy_loss * attention_mask).sum() / attention_mask.sum()


    masked_entropy = (entropy * attention_mask).sum() / attention_mask.sum()
    rl_loss = masked_policy_loss - entropy_coeff * masked_entropy

    values_new_clipped = torch.clamp(
        values_new,
        values_old - ppo_clip_epsilon,
        values_old + ppo_clip_epsilon
    )

    unclipped_value_loss = (values_new - returns) ** 2
    clipped_value_loss = (values_new_clipped - returns) ** 2
    value_loss = 0.5 * torch.mean(torch.max(unclipped_value_loss, clipped_value_loss))

    value_loss = value_loss_coeff * value_loss

    with torch.no_grad():
        # KL-divergence approximation between old and new policies
        kl_div = torch.mean((ratio - 1) - (log_probs_new - log_probs_old))

        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > ppo_clip_epsilon).float())

    metrics = {
        "kl_divergence": kl_div,
        "clip_fraction": clip_fraction,
        "policy_entropy": masked_entropy,
        "advantages_mean": advantages.mean(),
        "returns_mean": returns.mean(),
    }

    return rl_loss, value_loss, metrics