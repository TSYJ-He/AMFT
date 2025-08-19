# AMFT: Aligning LLM Reasoners by Meta-Learning the Optimal Imitation-Exploration Balance
This repository contains the official implementation for the paper: **"AMFT: Aligning LLM Reasoners by Meta-Learning the Optimal Imitation-Exploration Balance"**.

## Abstract

Large Language Models (LLMs) are typically fine-tuned for reasoning tasks through a two-stage pipeline of Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL), a process fraught with catastrophic forgetting and suboptimal trade-offs between imitation and exploration. Recent single-stage methods attempt to unify SFT and RL using heuristics, but lack a principled mechanism for dynamically balancing the two paradigms. In this paper, we reframe this challenge through the theoretical lens of **implicit rewards**, viewing SFT and RL not as distinct methods but as complementary reward signals. We introduce **Adaptive Meta Fine‑Tuning (AMFT)**, a novel single-stage algorithm that learns the optimal balance between SFT's implicit, path-level reward and RL's explicit, outcome-based reward. The core of AMFT is a **meta-gradient adaptive weight controller** that treats the SFT-RL balance as a learnable parameter, dynamically optimizing it to maximize long-term task performance. AMFT consistently establishes a new state-of-the-art and demonstrates superior generalization on out-of-distribution (OOD) tasks.

## Framework Overview
![Framework](https://github.com/user-attachments/assets/58a3c829-dfea-44b3-941d-9ca8fd169edb)

The core of AMFT is a single-stage training loop that dynamically and proactively learns the optimal balance between SFT (imitation) and RL (exploration) using a meta-gradient adaptive weight controller.

## Installation

1.  **Clone the repository**:


2.  **Create a Conda environment** (Recommended):
    ```bash
    conda create -n amft python=3.10
    conda activate amft
    ```

3.  **Install PyTorch**: Install from the official PyTorch website according to your CUDA version. For example, for CUDA 12.1:
    ```bash
    pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

4.  **Install other dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5.  **(Optional) Install vLLM**: For maximum efficiency in RL rollouts and evaluation, installing vLLM is recommended. Please refer to the [official documentation](https://docs.vllm.ai/en/latest/getting_started/installation.html) for installation instructions.

