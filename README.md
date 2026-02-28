# 🤖 miniGPT — JAX Implementation (Google Colab)

A minimal implementation of a **GPT-style Transformer language model** trained using the **JAX AI stack**, fully executed inside **Google Colab**.

This project focuses on understanding the **core internals of GPT models** by implementing a decoder-only Transformer from scratch using JAX’s functional programming style.

---

## ✨ Features

- 🧠 GPT-style **autoregressive language model**
- ⚡ Built with **JAX + Flax + Optax**
- 📓 **Single Google Colab notebook**
- 🔥 GPU / TPU compatible
- ✍️ Supports basic **text generation**
- 🎯 Designed for **learning & experimentation**

---

## 🧰 Tech Stack

| Tool | Usage |
|-----|------|
| 🧮 JAX | Numerical computation |
| 🧱 Flax (Linen) | Neural network layers |
| 🎛️ Optax | Optimizers |
| 📊 NumPy | Data handling |
| ☁️ Google Colab | Training environment |

---

## 📂 Project Structure

This repository intentionally uses **no folder structure**.

All implementation is contained in a single notebook:
miniGPT_JAX_Colab.ipynb

The notebook includes:
- 📥 Dataset loading & preprocessing  
- 🔤 Tokenization  
- 🧠 miniGPT model definition  
- 🔁 Training loop  
- 📉 Loss computation  
- ✨ Text generation  

---

## 🧠 Model Architecture
Tokens
↓
Token + Positional Embeddings
↓
N × Transformer Decoder Blocks
├─ Multi-Head Causal Self-Attention
├─ Layer Normalization
└─ Feed Forward Network (MLP)
↓
Linear Projection
↓
Next Token Prediction


✔ Uses **causal masking** for autoregressive training.

---

## ▶️ How to Run

1. 📓 Open `miniGPT_JAX_Colab.ipynb` in **Google Colab**
2. ⚙️ Enable **GPU / TPU**  
   `Runtime → Change runtime type`
3. 📦 Install dependencies:
   ```bash
   pip install -U jax jaxlib flax optax
