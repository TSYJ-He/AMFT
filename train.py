# /amft-project/train.py

import argparse
import yaml
import os
import torch
import logging
from typing import Dict, Any
from src.utils.logging import setup_logging
from src.model.policy_model import PolicyModel
from data.unified_datamodule import UnifiedDataModule
from src.trainer.amft_trainer import AMFTTrainer

logger = logging.getLogger(__name__)
def merge_configs(base_config: Dict[str, Any], cli_args: argparse.Namespace) -> Dict[str, Any]:

    merged_config = base_config.copy()
    cli_dict = vars(cli_args)
    for key, value in cli_dict.items():
        if value is not None and key in merged_config:
            merged_config[key] = value
            logger.info(f"Overriding config '{key}' with CLI value: {value}")

    return merged_config


def main(args):
    setup_logging()

    logger.info(f"Loading main configuration from: {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    config = merge_configs(config, args)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{os.getenv('LOCAL_RANK', 0)}")
    else:
        device = torch.device("cpu")

    torch.manual_seed(config['seed'])

    logger.info(f"Process rank: {os.getenv('RANK', 0)}, Device: {device}")
    logger.info(f"Starting training for task: {config['task_name']}")
    logger.info(f"Using model: {config['model_name_or_path']}")

    tokenizer = AutoTokenizer.from_pretrained(config['model_name_or_path'], trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_args = {'torch_dtype': torch.bfloat16}
    policy_model = PolicyModel(config['model_name_or_path'], model_args=model_args).to(device)

    data_module = UnifiedDataModule(config, tokenizer)
    data_module.setup()
    train_dataloader = data_module.get_train_dataloader()
    val_dataloader = data_module.get_val_dataloader()

    training_paradigm = args.training_paradigm

    trainer = None
    if training_paradigm == 'amft':
        logger.info("Initializing AMFTTrainer...")
        # Combine relevant config sections for the trainer
        trainer_config = {
            **config['training'],
            **config['rl_params'],
            'controller': config['controller'],
            'output_dir': config['output_dir']
        }
        trainer = AMFTTrainer(
            config=trainer_config,
            policy_model=policy_model,
            sft_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )
    # elif training_paradigm == 'sft_only':
    #     logger.info("Initializing SFTOlyTrainer...")
    #     # trainer = SFTOlyTrainer(...)
    # elif training_paradigm == 'rl_only':
    #     logger.info("Initializing RLOnlyTrainer...")
    #     # trainer = RLOnlyTrainer(...)
    else:
        raise ValueError(f"Unknown training paradigm: '{training_paradigm}'. "
                         "Please specify a valid paradigm (e.g., 'amft').")

    if trainer:
        logger.info(f"Starting training with the '{training_paradigm}' paradigm...")
        trainer.train()
        logger.info("Training complete.")
    else:
        logger.error("Trainer could not be initialized.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main script to run AMFT and baseline training.")

    parser.add_argument("--config_path", type=str, default="configs/main_config.yaml",
                        help="Path to the main YAML configuration file.")
    parser.add_argument("--training_paradigm", type=str, default="amft",
                        help="The training paradigm to use (e.g., 'amft', 'sft_only', 'rl_only').")

    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    main(args)