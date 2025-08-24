# AMFT: Aligning LLM Reasoners by Meta-Learning the Optimal Imitation-Exploration Balance

[![arXiv](https://img.shields.io/badge/arXiv-2508.06944-b31b1b.svg)](https://arxiv.org/abs/2508.06944) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3123/)

![Framework](https://github.com/user-attachments/assets/90b33fa6-66de-4141-b83e-8e07d9f5763d)


This repository contains the official implementation of **Adaptive Meta Fine-Tuning (AMFT)**, a novel single-stage algorithm for fine-tuning Large Language Models (LLMs) that dynamically balances Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) using meta-gradients. AMFT reframes SFT and RL as complementary reward signals and learns an optimal training curriculum to maximize long-term task performance, achieving state-of-the-art results on mathematical reasoning, abstract visual reasoning, and vision-language navigation benchmarks.

The code is based on the paper: [AMFT: Aligning LLM Reasoners by Meta-Learning the Optimal Imitation-Exploration Balance](https://arxiv.org/abs/2508.06944) (Preprint, under review, arXiv:2508.06944v1 [cs.LG] 9 Aug 2025).

If you find this work useful, please cite our paper (see [Citation](#citation) below).

## Key Features
- **Unified SFT-RL Framework**: Dynamically balances imitation (SFT) and exploration (RL) via a learnable parameter μ optimized with meta-gradients.
- **Meta-Gradient Controller**: Forward-looking optimization for long-term performance, regularized by policy entropy for stability.
- **Benchmarks Supported**: Mathematical reasoning (e.g., MATH500, AIME24), visual reasoning (General Points), and vision-language navigation (V-IRL).
- **Reproducibility**: Scripts for training, evaluation, and ablation studies to replicate paper results.

## Installation

### Prerequisites
- Python 3.12.3+
- CUDA-compatible GPU (recommended for training; tested on NVIDIA A100/V100)
- Basic libraries: See `requirements.txt` for full list.

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/TSYJ-He/AMFT.git
   cd AMFT
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or venv\Scripts\activate on Windows
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   This includes core libraries like PyTorch, Transformers, SymPy (for math rewards), and others used in the paper (e.g., NumPy, SciPy, Matplotlib for analysis).

   **Note**: No internet access is required during training beyond initial setup. All datasets are loaded from local paths or Hugging Face (cached).

## Quick Start

### Training AMFT
Run the main training script with default hyperparameters for mathematical reasoning:
```
python train_amft.py --model qwen2.5-math-7b --dataset openr1-math-46k --epochs 500 --batch_size 8 --lr 1e-5 --mu_init 0.5 --output_dir results/math
```
- `--model`: Base model (e.g., `qwen2.5-math-7b` or `llama-3.2-vision-11b`).
- `--dataset`: Path to dataset (e.g., OpenR1-Math-46k for math; General Points/V-IRL splits for visual tasks).
- `--mu_init`: Initial balance weight (default: 0.5).
- For full options, run `python train_amft.py --help`.

This will perform SFT warm-up, then enter the single-stage loop with meta-gradient updates.

### Evaluation
Evaluate a trained model on benchmarks:
```
python evaluate.py --model_path results/math/checkpoint-final --benchmark math500 --temperature 0.6 --max_length 8192
```
- Supports ID/OOD benchmarks as in the paper (e.g., MATH500, ARC-C, GPQA-D).
- Metrics: Accuracy (%) for math; Win/Success rates (%) for visual tasks.

### Example: Reproducing Math Results
To replicate Table 2 (Math Reasoning):
1. Download OpenR1-Math-46k dataset from [Hugging Face](https://huggingface.co/datasets/Elliott/Openr1-Math-46k-8192).
2. Train: `python train_amft.py --model qwen2.5-math-7b --dataset path/to/openr1-math --epochs 500`.
3. Evaluate on ID benchmarks: `python evaluate.py --model_path results/math/checkpoint-final --benchmark all_id`.
4. Compare with baselines (scripts in `baselines/`).

Expected runtime: ~10-20 hours on 4x A100 GPUs for full training.

### For more infomation please go to the appendix

<img width="3000" height="1800" alt="Appendix_GP_Dynamics" src="https://github.com/user-attachments/assets/bc9a8e0e-2ccd-45c2-8ce6-e30963074676" />
<img width="3000" height="1800" alt="Appendix_VIRL_Dynamics" src="https://github.com/user-attachments/assets/86d2a653-039a-4573-95e0-25682c8ba7ea" />


## Dependencies
Key libraries (from `requirements.txt`):
- `torch==2.0.1`
- `transformers==4.35.0`
- `sympy` (for math rewards)
- `matplotlib` (for plots)
- `numpy`, `scipy`, `pandas`

No additional pip installs are needed during runtime.

## Contributing
We welcome contributions! Please:
1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/new-thing`).
3. Commit changes (`git commit -am 'Add new thing'`).
4. Push to the branch (`git push origin feature/new-thing`).
5. Open a Pull Request.

For issues, use the GitHub Issues tab.

## Citation
If you use AMFT in your research, please cite:
```
@article{he2025amft,
  title={AMFT: Aligning LLM Reasoners by Meta-Learning the Optimal Imitation-Exploration Balance},
  author={Lixuan He and Jie Feng and Yong Li},
  journal={arXiv preprint arXiv:2508.06944},
  year={2025}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built on Qwen2.5-Math and LLaMA-3.2-Vision base models.
- Thanks to the authors of baselines (e.g., SRFT, LUFFY) for open-sourcing their work.
- Contact: helx23@mails.tsinghua.edu.cn for questions.

For more details, refer to the paper or appendices. Happy fine-tuning! 🚀




