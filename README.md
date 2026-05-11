# LLM From Scratch

A minimal GPT-style Transformer Language Model implemented completely from scratch using PyTorch.

This project was built to understand the internal workings of modern Large Language Models including:

- Tokenization
- Self Attention
- Multi-Head Attention
- Transformer Blocks
- Positional Encoding
- Autoregressive Text Generation
- Training and Evaluation Pipelines

---

# Features

- GPT-style decoder-only transformer
- Multi-head causal self-attention
- Sinusoidal positional encodings
- Custom training loop
- Text generation support
- Config-driven hyperparameters
- PyTorch implementation
- TikToken tokenizer support
- Model checkpoint saving/loading

---

# Project Structure

```text
.
├── train.py          # Main training script
├── eval.py           # Text generation / evaluation script
├── train.txt         # Training dataset
├── test.txt          # Prompt text for evaluation
├── vocab.txt         # Generated vocabulary
├── model-ckpt.pt     # Saved model checkpoint
└── README.md
