# /amft-project/src/model/policy_model.py

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from typing import Dict, Optional, List
class PolicyModel(nn.Module):
    def __init__(self, model_name_or_path: str, model_args: Dict = None):
        super().__init__()
        if model_args is None:
            model_args = {}
            
        self.model_name_or_path = model_name_or_path
        self.base_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_args,
            trust_remote_code=True # Necessary for models like Qwen
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.value_head = nn.Linear(self.base_model.config.hidden_size, 1, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
        }

        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values

        outputs = self.base_model(**model_inputs)
        
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1] # Get last layer hidden states

        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_token_hidden_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        
        value = self.value_head(last_token_hidden_states).squeeze(-1)

        return {"logits": logits, "value": value}
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        **generation_kwargs,
    ) -> List[int]:

        self.base_model.eval()
        
        prompt_len = inputs.get("input_ids").shape[1]
        
        generated_ids = self.base_model.generate(
            **inputs,
            **generation_kwargs,
        )

        return generated_ids[:, prompt_len:]

    def get_action_log_prob_and_entropy(
        self, 
        logits: torch.Tensor, 
        actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        action_logits = logits[:, :-1, :]
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_probs = torch.gather(log_probs, 2, actions.unsqueeze(-1)).squeeze(-1)
        probs = F.softmax(action_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        return {"log_prob": action_log_probs, "entropy": entropy}

    def get_base_model(self) -> PreTrainedModel:
        return self.base_model