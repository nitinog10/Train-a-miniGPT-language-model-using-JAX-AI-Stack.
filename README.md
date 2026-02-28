# Train-a-miniGPT-language-model-using-JAX-AI-Stack
🧠 miniGPT Training using JAX (Google Colab)

This repository contains a miniGPT-style Transformer Language Model implemented and trained entirely inside a single Google Colab notebook using the JAX AI stack.

The goal of this project is to demonstrate how a GPT-like autoregressive language model can be built from scratch using JAX’s functional programming paradigm.

🚀 Project Highlights

✅ Single-notebook implementation (Colab-friendly)

🧠 Decoder-only Transformer (miniGPT)

⚡ Built using JAX + Flax + Optax

🔥 GPU/TPU compatible via Google Colab

📚 Educational & minimal implementation

🛠️ Tech Stack

JAX – High-performance numerical computing

Flax (Linen) – Neural network layers

Optax – Optimizers & learning-rate schedules

NumPy – Data handling

Google Colab – Training environment

📓 Notebook

All code lives in one notebook:

miniGPT_JAX_Colab.ipynb

The notebook includes:

Dataset loading & preprocessing

Tokenization

Transformer (miniGPT) model definition

Training loop

Loss visualization

Text generation

🧠 Model Architecture (miniGPT)

Token Embedding

Positional Embedding

Stacked Transformer Decoder Blocks

Multi-Head Self Attention (Causal)

Layer Normalization

Feed Forward Network (MLP)

Linear Output Projection

Causal masking is used to ensure autoregressive behavior.

▶️ How to Run (Google Colab)
1️⃣ Open in Colab

Upload or open miniGPT_JAX_Colab.ipynb in Google Colab

2️⃣ Enable Accelerator
Runtime → Change runtime type → GPU / TPU
3️⃣ Install Dependencies
!pip install -U jax jaxlib flax optax

(No CUDA setup required on Colab)

4️⃣ Run All Cells

The notebook is fully sequential — just Run All.

📈 Training Details

Objective: Causal Language Modeling

Optimizer: Adam / AdamW (Optax)

Loss: Cross-Entropy

Precision: float32

Device: GPU / TPU (Colab)

✨ Text Generation

After training, the model can generate text from a prompt:

generate(
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.8
)
📌 Key Learnings

GPT architecture fundamentals

Self-attention & causal masking

JAX functional training loops

Efficient model training in Colab

🔮 Possible Improvements

Add BPE / SentencePiece tokenizer

Save & load checkpoints

Scale model size

Convert notebook into modular repo

Serve inference via API

📜 License

This project is licensed under the MIT License.

🙌 Acknowledgements

OpenAI GPT architecture

JAX, Flax & Optax teams

Google Colab for compute support
