# Copy code from chapter-4.ipynb

import torch
import torch.nn as nn
# import code from 
from previous_chapters_two_three import MultiHeadAttention


# Chapter 4: Layer normalization to normalize activations (mimics GPT-2 implementation)
class LayerNorm(nn.Module):
  def __init__(self, emb_dim):
    super().__init__()
    self.eps = 1e-5 # 0.00001
    self.scale = nn.Parameter(torch.ones(emb_dim))
    self.shift = nn.Parameter(torch.zeros(emb_dim))

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False) 
    # unbiased=False means we use the population variance to mimic the gpt2 implementation
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    return self.scale * norm_x + self.shift


# Chapter 4: GELU activation function using approximation to mimic GPT-2
class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    # we are using the approximation of the GELU activation function to mimic the gpt2 implementation
    return 0.5 * x * (1 + torch.tanh(
      torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
      (x + 0.044715 * torch.pow(x, 3.0))
    ))


# Chapter 4: Feed-forward network with GELU activation (4x expansion like GPT-2)
class FeedForward(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(cfg["embed_dim"], 4 * cfg["embed_dim"]), # 4 is what gpt2 uses
      GELU(),
      nn.Linear(4 * cfg["embed_dim"], cfg["embed_dim"])
    )

  def forward(self, x):
    return self.layers(x)


# Chapter 4: Transformer block combining attention, feed-forward, layer norm, and shortcut connections
class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.attn = MultiHeadAttention(
      d_in = cfg["embed_dim"],
      d_out = cfg["embed_dim"],
      context_length = cfg["context_length"],
      num_heads = cfg["n_heads"],
      dropout = cfg["drop_rate"],
      qkv_bias = cfg["qkv_bias"]
    )

    self.ff = FeedForward(cfg)
    self.norm1 = LayerNorm(cfg["embed_dim"])
    self.norm2 = LayerNorm(cfg["embed_dim"])
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

  def forward(self, x):
    shortcut = x
    x = self.norm1(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut

    shortcut = x # keep the x from the previous layer for the shortcut connection
    x = self.norm2(x)
    x = self.attn(x)
    x = self.drop_shortcut(x)
    x = x + shortcut # add the shortcut connection

    return x


# Chapter 4: Complete GPT model architecture with embeddings, transformer blocks, and output head
class GPTModel(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
    self.pos_emb = nn.Embedding(cfg["context_length"], cfg["embed_dim"])
    self.dropout = nn.Dropout(cfg["drop_rate"])

    # Placeholder for the transformer blocks
    self.trf_blocks = nn.Sequential(
      *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
    )

    # Placeholder for layer norm
    self.final_norm = LayerNorm(cfg["embed_dim"])
    self.out_head = nn.Linear(
      cfg["embed_dim"], cfg["vocab_size"], bias=False
    )

  def forward(self, in_idx):
    batch_size, seq_len = in_idx.shape
    tok_embeds = self.tok_emb(in_idx)
    pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
    x = tok_embeds + pos_embeds
    x = self.dropout(x)
    x = self.trf_blocks(x)
    x = self.final_norm(x)
    logits = self.out_head(x)
    return logits


# Chapter 4: Simple text generation function to generate text using the GPT model
def generate_text_simple(model, idx, max_new_tokens, context_size):
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:] # truncating to the size the model supports (if it was longer)

    with torch.no_grad(): # we are not training, so we do not need to compute gradients
      logits = model(idx_cond)
    logits = logits[:, -1, :] # only the last row of the logits (the new token)

    probas = torch.softmax(logits, dim=-1) # computing the probabilities
    idx_next = torch.argmax(probas, dim=-1, keepdim=True) # argmax looks up the index position (finding the index position with the highest probability)

    idx = torch.cat((idx, idx_next), dim=1) # concatenating the new token to the context

  return idx