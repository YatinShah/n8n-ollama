import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import math
import torch.nn as nn
from torch.nn import functional as F

# --- Model Architecture Definition (Must match the training script) ---
# We need to redefine the model architecture classes exactly as they were in train_slm.py
# for torch.load_state_dict to work correctly.

class Head(nn.Module):
    def __init__(self, head_size, N_EMBD, BLOCK_SIZE, DROPOUT):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)
        self.BLOCK_SIZE = BLOCK_SIZE

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, N_EMBD, BLOCK_SIZE, DROPOUT):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, N_EMBD, BLOCK_SIZE, DROPOUT) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, N_EMBD, DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, N_EMBD, N_HEAD, BLOCK_SIZE, DROPOUT):
        super().__init__()
        head_size = N_EMBD // N_HEAD
        self.sa = MultiHeadAttention(N_HEAD, head_size, N_EMBD, BLOCK_SIZE, DROPOUT)
        self.ffwd = FeedForward(N_EMBD, DROPOUT)
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SimpleSLM(nn.Module):
    def __init__(self, vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT):
        super().__init__()
        self.BLOCK_SIZE = BLOCK_SIZE
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD, BLOCK_SIZE, DROPOUT) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        # Loss calculation omitted for API inference endpoint

        return logits, None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for step in range(max_new_tokens):
            # Crop idx to the last BLOCK_SIZE tokens (to handle memory limits)
            idx_cond = idx[:, -self.BLOCK_SIZE:]
            # Get predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # --- DEBUGGING INFO START (FIXED) ---
            if True or step % 10 == 0 or step == max_new_tokens - 1:
                # Get top 5 probabilities/indices (for the 0-th item in the batch)
                top_probs, top_indices = torch.topk(probs[0], 5) 
                
                # Decode indices to characters (i is already a tensor here)
                top_chars = [itos[i.item()] for i in top_indices]

                # FIX IS HERE: idx[0, :].tolist() returns a list of ints. We should iterate over ints.
                context_list_of_ints = idx[0, :].tolist()
                current_context_chars = "".join([itos[i] for i in context_list_of_ints])[-30:]
                
                print(f"\n[DEBUG Step {step+1}/{max_new_tokens}] Context suffix: '{current_context_chars}'")
                print(f"  Top 5 Predictions:")
                for char, prob in zip(top_chars, top_probs):
                    display_char = repr(char) if len(char) > 1 or char in ('\n', '\t') else char
                    print(f"    Char: {display_char}, Prob: {prob.item()*100:.2f}%")
            # --- DEBUGGING INFO END ---

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# ----------------------------------------------------------------------
# FastAPI Application
# ----------------------------------------------------------------------

app = FastAPI(
    title="Custom SLM Inference API",
    description="An API for generating text using a locally trained, CPU-only SLM model.",
)

# Global variables to store the loaded model and tokenizers
model = None
stoi, itos = None, None
VOCAB_SIZE, BLOCK_SIZE = 0, 0
DEVICE = torch.device("cpu")
MODEL_PATH = './output_models/custom_slm_model_cpu.pth'

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100

class GenerateResponse(BaseModel):
    generated_text: str

def load_model_and_metadata():
    """Loads the trained model and associated vocabulary/config."""
    global model, stoi, itos, VOCAB_SIZE, BLOCK_SIZE

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    config = checkpoint['config']
    VOCAB_SIZE = checkpoint['vocab_size']
    BLOCK_SIZE = config['BLOCK_SIZE']
    chars = checkpoint['chars']
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    model = SimpleSLM(
        vocab_size=VOCAB_SIZE,
        N_EMBD=config['N_EMBD'],
        N_HEAD=config['N_HEAD'],
        N_LAYER=config['N_LAYER'],
        BLOCK_SIZE=BLOCK_SIZE,
        DROPOUT=config['DROPOUT']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH}")
    print(f"Model config: {config}, Vocab size: {VOCAB_SIZE}")


@app.on_event("startup")
async def startup_event():
    """Load the model when the FastAPI application starts up."""
    load_model_and_metadata()


def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])


@app.post("/generate", response_model=GenerateResponse)
async def generate_text_api(request: GenerateRequest):
    """
    Generates text based on a provided prompt using the custom SLM.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not yet loaded or available.")
    
    context = torch.tensor(encode(request.prompt), dtype=torch.long, device=DEVICE)[None, ...]
    
    # Generate new tokens
    generated_indices = model.generate(context, max_new_tokens=request.max_tokens).tolist()
    
    # Decode and return the full text (including the prompt)
    full_generated_text = decode(generated_indices[0])
    
    return GenerateResponse(generated_text=full_generated_text)

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify API is running and model is loaded.
    """
    status = "Model Loaded" if model is not None else "Model Loading Error/Not Found"
    return {"status": "ok", "model_status": status, "device": str(DEVICE)}

