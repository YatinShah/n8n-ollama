import torch
import torch.nn as nn
from torch.nn import functional as F
import PyPDF2
import os
import numpy as np
from tqdm import tqdm
import math

# --- Configuration ---
# Set device explicitly to CPU
DEVICE = torch.device("cpu")
BATCH_SIZE = 16
BLOCK_SIZE = 64  # Max context length for predictions
MAX_ITERS = 1000  # Limited iterations for a quick, local example
EVAL_INTERVAL = 100
LEARNING_RATE = 1e-3
N_EMBD = 64
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1
DATA_DIR = './input_pdfs/'
MODEL_PATH = './output_models/custom_slm_model_cpu.pth' 

print(f"--- Configuration ---")
print(f"Device set to: {DEVICE}")
print(f"Using BATCH_SIZE: {BATCH_SIZE}, BLOCK_SIZE: {BLOCK_SIZE}")
print(f"Saving model to: {MODEL_PATH}")

# ----------------------------------------------------------------------
# 1. Data Processing and Tokenization
# ----------------------------------------------------------------------

def extract_text_from_pdfs(directory):
    """Reads all PDF files in a directory and concatenates the text."""
    full_text = ""
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            print(f"Processing PDF: {filepath}")
            try:
                with open(filepath, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n"
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    if not full_text:
        raise ValueError(f"No text extracted from PDFs in {directory}. Please check your input files.")
    return full_text

# Get the raw data
text = extract_text_from_pdfs(DATA_DIR)
print(f"\nTotal length of corpus: {len(text)} characters.")

# Create vocabulary and tokenizers (character level for simplicity)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Tokenize the entire text and convert to a PyTorch tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(f"Data tensor shape: {data.shape}")

# Split data into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """Generates a small batch of data of inputs x and targets y"""
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_source[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_source[i+1:i+BLOCK_SIZE+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# ----------------------------------------------------------------------
# 2. Model Definition (Minimal Transformer)
# ----------------------------------------------------------------------

# Helper components for the Transformer
class Head(nn.Module):
    """One self-attention head"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        # Register buffer for the mask; not a model parameter
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores ("affinities")
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1]**-0.5) # (B, T, T)
        # Mask future information (decoder block)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        # Concatenate results of all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD), # Projection layer
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self):
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.sa = MultiHeadAttention(N_HEAD, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        # Apply residual connections and layer normalization
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleSLM(nn.Module):
    def __init__(self):
        super().__init__()
        # Token embeddings table
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        # Positional embeddings table
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        # The sequence of Transformer blocks
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        # Final layer normalization
        self.ln_f = nn.LayerNorm(N_EMBD)
        # Linear layer to project embeddings back to vocabulary logits
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos = torch.arange(T, device=DEVICE) # (T)
        pos_emb = self.position_embedding_table(pos) # (T, C)
        x = tok_emb + pos_emb # (B, T, C) combines token and position info
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Reshape for PyTorch cross_entropy calculation
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last BLOCK_SIZE tokens (to handle memory limits)
            idx_cond = idx[:, -BLOCK_SIZE:]
            # Get predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Initialize the model and move to CPU
model = SimpleSLM()
m = model.to(DEVICE)
print(f"Model initialized on {DEVICE}, total parameters: {sum(p.numel() for p in m.parameters())/1e6:.2f} Million")

# ----------------------------------------------------------------------
# 3. Training Loop and Export
# ----------------------------------------------------------------------

# Create a PyTorch optimizer (AdamW)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_INTERVAL)
        for k in range(EVAL_INTERVAL):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print("\nStarting training loop...")
for iter in tqdm(range(MAX_ITERS)):

    # Evaluate loss occasionally
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Forward pass: evaluate the loss
    logits, loss = model(xb, yb)
    
    # Backward pass: zero gradients, backpropagate, update weights
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training finished.")

# ----------------------------------------------------------------------
# 4. Model Export
# ----------------------------------------------------------------------

# Save the model state dictionary and metadata needed for reloading
model_export_data = {
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'chars': chars,
    'config': {
        'N_EMBD': N_EMBD, 'N_HEAD': N_HEAD, 'N_LAYER': N_LAYER,
        'BLOCK_SIZE': BLOCK_SIZE, 'DROPOUT': DROPOUT
    }
}
torch.save(model_export_data, MODEL_PATH)
print(f"Model successfully saved to {MODEL_PATH}")

# ----------------------------------------------------------------------
# 5. Demonstration/Inference (Optional)
# ----------------------------------------------------------------------

print("\n--- Demonstration of generated text ---")
# Start generation with a simple newline character to prompt a start
context = torch.tensor(encode('\n'), dtype=torch.long, device=DEVICE)[None, ...]
# Generate 200 new tokens
generated_indices = m.generate(context, max_new_tokens=200)[0].tolist()
print(decode(generated_indices))

