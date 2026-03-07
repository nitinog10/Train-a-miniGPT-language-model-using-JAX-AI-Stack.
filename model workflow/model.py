```python
# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
import datasets
from transformers import AutoTokenizer
import tensorflow as tf
from tqdm.auto import tqdm

# Configuration class for the model
class MiniGPTConfig:
    vocab_size: int = 50257  # GPT-2 vocabulary size
    max_position_embeddings: int = 128  # Max context length
    hidden_size: int = 768    # Embedding dimension
    num_attention_heads: int = 12 # Number of attention heads
    num_hidden_layers: int = 6 # Number of transformer blocks
    intermediate_size: int = 3072 # Size of the feed-forward network

# Transformer Block class
class TransformerBlock(nn.Module):
    config: MiniGPTConfig

    @nn.compact
    def __call__(self, inputs, attention_mask):
        normed_inputs = nn.LayerNorm()(inputs)
        attention_bias = nn.make_attention_mask(
            attention_mask > 0, attention_mask > 0, dtype=jnp.float32
        )
        attn_output = nn.SelfAttention(
            num_heads=self.config.num_attention_heads,
            qkv_features=self.config.hidden_size,
            name='self_attention',
        )(normed_inputs, mask=attention_bias)
        attn_output = inputs + attn_output
        normed_attn_output = nn.LayerNorm()(attn_output)
        ffn_output = nn.Dense(self.config.intermediate_size, name='wi')(normed_attn_output)
        ffn_output = nn.gelu(ffn_output)
        ffn_output = nn.Dense(self.config.hidden_size, name='wo')(ffn_output)
        output = attn_output + ffn_output
        return output

# MiniGPT model class
class MiniGPT(nn.Module):
    config: MiniGPTConfig

    @nn.compact
    def __call__(self, input_ids, attention_mask):
        token_embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            name='token_embeddings',
        )(input_ids)
        position_ids = jnp.arange(input_ids.shape[-1], dtype=jnp.int32)
        position_embeddings = nn.Embed(
            num_embeddings=self.config.max_position_embeddings,
            features=self.config.hidden_size,
            name='position_embeddings',
        )(position_ids)
        embeddings = token_embeddings + position_embeddings
        x = embeddings
        for i in range(self.config.num_hidden_layers):
            x = TransformerBlock(self.config, name=f'transformer_block_{i}')(x, attention_mask)
        x = nn.LayerNorm(name='final_norm')(x)
        logits = nn.Dense(
            features=self.config.vocab_size,
            name='output_logits',
            kernel_init=nn.initializers.normal(stddev=0.02)
        )(x)
        return logits

# Training step function
@jax.jit(static_argnames=['config'])
def train_step(state: TrainState, batch: dict, dropout_rng: jax.Array, config: MiniGPTConfig):
    def loss_fn(params):
        labels = batch['labels']
        logits = state.apply_fn({'params': params}, input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], rngs={'dropout': dropout_rng})
        one_hot_labels = jax.nn.one_hot(labels, num_classes=config.vocab_size)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
        mask = batch['attention_mask'].reshape(loss.shape)
        loss = loss * mask
        loss = jnp.sum(loss) / jnp.sum(mask)
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

# Main training function
def main():
    # Load dataset and tokenizer
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize and preprocess dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=config.max_position_embeddings)
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // config.max_position_embeddings) * config.max_position_embeddings
        result = {
            k: [t[i : i + config.max_position_embeddings] for i in range(0, total_length, config.max_position_embeddings)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000, num_proc=1)

    # Define model and optimizer
    config = MiniGPTConfig()
    model = MiniGPT(config)
    dummy_input_ids = jax.random.randint(jax.random.PRNGKey(0), (1, config.max_position_embeddings), 0, config.vocab_size)
    dummy_attention_mask = jnp.ones_like(dummy_input_ids)
    key = jax.random.PRNGKey(1)
    params = model.init(key, dummy_input_ids, dummy_attention_mask)
    learning_rate = 1e-4
    warmup_steps = 1000
    cooldown_steps = 10000
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=cooldown_steps,
        end_value=learning_rate * 0.01
    )
    optimizer = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.95, eps=1e-8)
    state = TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        tx=optimizer
    )

    # Training loop
    num_epochs = 3
    batch_size = 16
    checkpoint_dir = './checkpoints'
    key = jax.random.PRNGKey(0)
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        epoch_train_dataset = lm_datasets['train'].shuffle().to_tf_dataset(
            columns=['input_ids', 'attention_mask', 'labels'],
            shuffle=False,
            batch_size=batch_size,
            collate_fn=None,
            drop_remainder=True
        )
        key, dropout_rng = jax.random.split(key)
        for batch_idx, batch_tf in enumerate(tqdm(epoch_train_dataset, desc=f"Epoch {epoch+1} Batches")):
            batch = {k: jnp.asarray(v) for k, v in batch_tf.items()}
            dropout_rng, new_dropout_rng = jax.random.split(dropout_rng)
            state, loss = train_step(state, batch, new_dropout_rng, config)
            if batch_idx % 10 == 0:
                tqdm.write(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.4f}")
        checkpoints.save_checkpoint(ckpt_dir=checkpoint_dir, target=state, step=epoch, overwrite=True, keep=3)
        tqdm.write(f"Saved checkpoint for epoch {epoch} to {checkpoint_dir}")
    print("Training complete.")

if __name__ == "__main__":
    main()
```