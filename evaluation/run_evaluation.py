# /amft-project/evaluation/run_evaluation.py

import torch
import argparse
import yaml
import logging
import os
from tqdm import tqdm
from typing import Dict, Any, List
from src.model.policy_model import PolicyModel
from data.unified_datamodule import UnifiedDataModule
from src.utils.logging import setup_logging
from src.utils.checkpointing import load_checkpoint
class OATGraderWrapper:
    def __init__(self, task_name: str):
        self.task_name = task_name
        logger.info(f"Initialized Grader for task: {self.task_name} (mocked).")
    def grade(self, prompts: List[str], generated_texts: List[str], ground_truths: List[str]) -> List[float]:
        scores = []
        for gen, gt in zip(generated_texts, ground_truths):

            try:
                gen_answer = self._extract_answer(gen)
                gt_answer = self._extract_answer(gt)
                if gen_answer is not None and gt_answer is not None and gen_answer == gt_answer:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            except Exception:
                scores.append(0.0)
        return scores

    def _extract_answer(self, text: str) -> Optional[str]:
        match = re.search(r'\\boxed\{(.+?)\}', text)
        if match:
            return match.group(1).strip()

        numbers = re.findall(r'\d+', text)
        return numbers[-1] if numbers else None


def evaluate(args):

    setup_logging()
    logger = logging.getLogger(__name__)

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)

    model_args = {'torch_dtype': torch.bfloat16}
    policy_model = PolicyModel(args.checkpoint_path, model_args=model_args).to(device)
    policy_model.eval()
    logger.info(f"Loaded model and tokenizer from {args.checkpoint_path}")

    eval_task_config = config[args.task_name]
    if args.eval_rule_variant:
        eval_task_config['rule_variant'] = args.eval_rule_variant
    if args.eval_visual_domain:
        eval_task_config['visual_domain'] = args.eval_visual_domain

    config[args.task_name] = eval_task_config  # Update main config
    config['task_name'] = args.task_name

    data_module = UnifiedDataModule(config, tokenizer)
    data_module.setup(stage='test')
    test_dataloader = data_module.get_val_dataloader()

    grader = OATGraderWrapper(task_name=args.task_name)
    results = []
    total_correct = 0
    total_samples = 0

    for batch in tqdm(test_dataloader, desc=f"Evaluating on {args.task_name}"):
        input_ids = batch['input_ids'].to(device)

        labels = batch['labels']

        generated_ids = policy_model.generate(
            inputs={'input_ids': input_ids, 'attention_mask': attention_mask},
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        labels[labels == -100] = tokenizer.pad_token_id
        ground_truth_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        prompts_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        scores = grader.grade(prompts_texts, generated_texts, ground_truth_texts)

        total_correct += sum(scores)
        total_samples += len(scores)

        for i in range(len(prompts_texts)):
            results.append({
                "prompt": prompts_texts[i],
                "generated_text": generated_texts[i],
                "ground_truth": ground_truth_texts[i],
                "score": scores[i]
            })

    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    logger.info("--- Evaluation Finished ---")
    logger.info(f"Task: {args.task_name}")
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Total Samples: {total_samples}")
    logger.info(f"Accuracy: {accuracy:.2f}%")

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Detailed results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on a trained AMFT model.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the model checkpoint directory.")
    parser.add_argument("--config_path", type=str, default="configs/main_config.yaml",
                        help="Path to the main configuration file.")
    parser.add_argument("--task_name", type=str, required=True,
                        help="Name of the task to evaluate (e.g., 'math_reasoning', 'general_points').")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional path to save detailed results in JSONL format.")

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling probability.")

    # OOD evaluation parameters
    parser.add_argument("--eval_rule_variant", type=str, default=None,
                        help="Specify OOD rule for GP/V-IRL (e.g., '11_12_13').")
    parser.add_argument("--eval_visual_domain", type=str, default=None,
                        help="Specify OOD domain for V-IRL/GP (e.g., 'worldwide', 'red').")

    args = parser.parse_args()
    evaluate(args)