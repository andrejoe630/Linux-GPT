import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# Hyperparameters (must match training)
block_size = 64
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2
device = "cpu"

# GPT-2 tokenizer
vocab_size = 50304
enc = tiktoken.get_encoding("gpt2")


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    """Full Transformer language model"""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature: float = 1.0, top_k: int = 50):
        """Generate new tokens given a context with temperature and top-k sampling."""
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # Get predictions
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)

            # Greedy decoding if temperature == 0
            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
            else:
                # Apply temperature
                logits = logits / temperature

                # Top-k filtering
                if top_k is not None and top_k > 0 and top_k < logits.size(-1):
                    v, _ = torch.topk(logits, top_k)
                    threshold = v[..., -1, None]  # (B, 1)
                    logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)

                # Convert to probabilities
                probs = F.softmax(logits, dim=-1)  # (B, C)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


if __name__ == "__main__":
    # Load trained model weights
    model = BigramLanguageModel().to(device)
    state_dict = torch.load("linux_gpt_v2.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print("Token-level Linux GPT chat with temperature & top-k sampling.")
    print("Type 'quit' or 'exit' as the prompt to stop.\n")

    while True:
        # Ask for temperature
        temp_str = input("Temperature (0 = greedy, default 1.0): ").strip()
        if temp_str == "":
            temperature = 1.0
        else:
            try:
                temperature = float(temp_str)
                if temperature < 0:
                    print("Temperature must be >= 0. Using 1.0.")
                    temperature = 1.0
            except ValueError:
                print("Could not parse temperature. Using 1.0.")
                temperature = 1.0

        # Ask for the prompt
        user_input = input("Prompt (or 'quit'/'exit'): ")
        if user_input.strip().lower() in {"quit", "exit"}:
            break

        # Encode user input to token IDs
        input_ids = enc.encode(user_input)
        if len(input_ids) == 0:
            continue

        idx = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            # Generate continuation with temperature and top-k
            out_ids = model.generate(
                idx,
                max_new_tokens=200,
                temperature=temperature,
                top_k=50,
            )[0].tolist()

        # Strip the prompt tokens so we only decode new tokens
        response_ids = out_ids[len(input_ids):]
        response_text = enc.decode(response_ids)

        print("Bot:", response_text)
