.

🧠 miniGPT — Training a GPT-Style Language Model with JAX (Colab)

A from-scratch implementation of a GPT-style Transformer Language Model, trained entirely in Google Colab using the JAX AI Stack.

This project demonstrates how modern large language models work internally by building a miniGPT using JAX, Flax, and Optax — without hiding complexity behind frameworks.
Everything runs inside one Google Colab notebook, making the project accessible, reproducible, and educational.

🌟 Why This Project Stands Out
# miniGPT (JAX, Google Colab)

This repository contains a miniGPT (GPT-style Transformer language model) implemented and trained using the JAX AI stack.  
The entire implementation is done inside a single Google Colab notebook.

The goal of this project is to understand and implement a decoder-only Transformer model for causal language modeling using JAX.

---

## Overview

- GPT-style autoregressive language model
- Decoder-only Transformer architecture
- Implemented using JAX, Flax, and Optax
- Trained on text data using causal language modeling
- Executed completely in Google Colab

---

## Technologies Used

- JAX
- Flax (Linen)
- Optax
- NumPy
- Google Colab

---

## Project Structure

This project does not follow a folder-based structure.

All code is contained in a single notebook:

The notebook includes:
- Dataset loading and preprocessing
- Tokenization
- Model definition (miniGPT)
- Training loop
- Loss computation
- Text generation

---

## Model Architecture

- Token Embedding
- Positional Embedding
- Multiple Transformer Decoder Blocks
  - Multi-Head Self-Attention (causal)
  - Layer Normalization
  - Feed Forward Network
- Linear output projection

Causal masking is used to ensure autoregressive behavior.

---

## Running the Notebook

1. Open `miniGPT_JAX_Colab.ipynb` in Google Colab
2. Enable GPU or TPU from Runtime settings
3. Install dependencies:
   ```bash
   pip install jax jaxlib flax optax
✅ From Scratch — No HuggingFace Trainer, no shortcuts
✅ JAX-Native — Pure functional training loops
✅ Single-Notebook Design — Clean & reproducible
✅ Autoregressive GPT Architecture
✅ GPU / TPU Ready (Colab)
✅ Ideal for Learning, Research & Interviews

🔥 What This Project Is

A decoder-only Transformer (GPT-style)

Trained using causal language modeling

Capable of text generation

Implemented in ~clean, readable JAX code

Designed to teach how GPT actually works

🧰 Tech Stack (JAX AI Stack)
Component	Purpose
JAX	High-performance numerical computing
Flax (Linen)	Neural network layers
Optax	Optimizers & learning rate schedules
NumPy	Data preprocessing
Google Colab	Training environment
📓 Notebook-Only Design

This repository intentionally contains a single notebook:

miniGPT_JAX_Colab.ipynb

Why?

No environment setup pain

No dependency hell

Fully reproducible in minutes

Perfect for demos, learning & hackathons

🧠 Model Architecture (miniGPT)
Input Tokens
     ↓
Token Embedding + Positional Embedding
     ↓
N × Transformer Decoder Blocks
     ├─ Multi-Head Causal Self-Attention
     ├─ Layer Normalization
     └─ Feed Forward Network (MLP)
     ↓
Linear Projection
     ↓
Next Token Prediction

✔ Causal masking ensures true autoregressive behavior

🚀 Getting Started (Google Colab)
1️⃣ Open the Notebook

Upload or open miniGPT_JAX_Colab.ipynb in Google Colab

2️⃣ Enable Hardware Acceleration
Runtime → Change runtime type → GPU or TPU
3️⃣ Install Dependencies
!pip install -U jax jaxlib flax optax
4️⃣ Run All Cells

That’s it. Training starts immediately.

📊 Training Details

Objective: Causal Language Modeling

Loss Function: Cross-Entropy

Optimizer: Adam / AdamW (Optax)

Precision: float32

Execution: JIT-compiled with JAX

Hardware: GPU / TPU (Colab)

✨ Text Generation

After training, the model can generate text autoregressively:

generate(
    prompt="Once upon a time",
    max_tokens=120,
    temperature=0.8
)

The output improves as training progresses.

📈 What You Learn From This Project

How GPT models work internally

Transformer attention mechanics

Causal masking implementation

Functional training loops in JAX

Performance optimization using jit & vmap

🎓 Who This Is For

AI / ML students

Researchers exploring JAX

Hackathon participants

Engineers preparing for LLM interviews

Anyone curious how GPT actually works

🔮 Future Extensions

Byte Pair Encoding (BPE / SentencePiece)

Checkpoint saving & loading

Larger GPT variants

TPU-optimized pipelines

REST API inference server

Modular repo version

📜 License

Licensed under the MIT License — free to use, modify, and distribute.

🙌 Acknowledgements

GPT architecture (OpenAI)

JAX, Flax & Optax teams

Google Colab for accessible compute
