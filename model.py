import os
import requests
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================
# CONFIG
# ============================================================

config = {
    "batch_size": 4,
    "context_length": 16,
    "d_model": 64,
    "num_blocks": 8,
    "num_heads": 4,
    "learning_rate": 1e-3,
    "dropout": 0.1,
    "max_iters": 5000,
    "eval_interval": 50,
    "eval_iters": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "TORCH_SEED": 1337,
}

torch.manual_seed(config["TORCH_SEED"])

device = config["device"]

# ============================================================
# LOAD TRAINING DATA
# ============================================================

vocab = set()

if not os.path.exists("train.txt"):
    url = "https://huggingface.co/datasets/Wannita/PyCoder/resolve/main/train.txt"

    with open("train.txt", "w") as f:
        f.write(requests.get(url).text)

with open("train.txt", "r", encoding="utf-8") as f:
    text = f.read()

if not os.path.exists("vocab.txt"):

    with open("train.txt", "r", encoding="utf-8") as f:
        text = f.read()
        vocab.update(text.split())

    with open("vocab.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(sorted(vocab)))

    print("Saved vocabulary to vocab.txt")

# ============================================================
# TOKENIZATION
# ============================================================

encoding = tiktoken.get_encoding("p50k_base")

tokenized_text = encoding.encode(text)

max_token_value = max(tokenized_text) + 1

tokenized_text = torch.tensor(
    tokenized_text,
    dtype=torch.long,
    device=device
)

# ============================================================
# TRAIN / VALID SPLIT
# ============================================================

split_idx = int(len(tokenized_text) * 0.9)

train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]

# ============================================================
# FEED FORWARD NETWORK
# ============================================================


class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.ffn = nn.Sequential(
            nn.Linear(
                in_features=config["d_model"],
                out_features=config["d_model"] * 4
            ),

            nn.ReLU(),

            nn.Linear(
                in_features=config["d_model"] * 4,
                out_features=config["d_model"]
            ),

            nn.Dropout(config["dropout"]),
        )

    def forward(self, x):
        return self.ffn(x)

# ============================================================
# ATTENTION
# ============================================================


class Attention(nn.Module):

    def __init__(self, head_size: int):
        super().__init__()

        self.head_size = head_size

        self.key_layer = nn.Linear(
            in_features=config["d_model"],
            out_features=self.head_size,
            bias=False
        )

        self.query_layer = nn.Linear(
            in_features=config["d_model"],
            out_features=self.head_size,
            bias=False
        )

        self.value_layer = nn.Linear(
            in_features=config["d_model"],
            out_features=self.head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    (
                        config["context_length"],
                        config["context_length"]
                    )
                )
            )
        )

        self.dropout_layer = nn.Dropout(config["dropout"])

    def forward(self, x):

        B, T, C = x.shape

        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)

        weights = (q @ k.transpose(-2, -1)) * (
            1.0 / math.sqrt(k.size(-1))
        )

        weights = weights.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(weights, dim=-1)

        weights = self.dropout_layer(weights)

        out = weights @ v

        return out

# ============================================================
# MULTI HEAD ATTENTION
# ============================================================


class MultiHeadAttention(nn.Module):

    def __init__(self, head_size: int):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                Attention(head_size=head_size)
                for _ in range(config["num_heads"])
            ]
        )

        self.projection_layer = nn.Linear(
            in_features=config["d_model"],
            out_features=config["d_model"]
        )

        self.dropout_layer = nn.Dropout(config["dropout"])

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)

        out = self.projection_layer(out)

        out = self.dropout_layer(out)

        return out

# ============================================================
# TRANSFORMER BLOCK
# ============================================================


class TransformerBlock(nn.Module):

    def __init__(self, num_heads: int):
        super().__init__()

        head_size = config["d_model"] // num_heads

        self.multi_head_attention_layer = MultiHeadAttention(
            head_size=head_size
        )

        self.feed_forward_layer = FeedForward()

        self.layer_norm_1 = nn.LayerNorm(config["d_model"])

        self.layer_norm_2 = nn.LayerNorm(config["d_model"])

    def forward(self, x):

        x = x + self.multi_head_attention_layer(
            self.layer_norm_1(x)
        )

        x = x + self.feed_forward_layer(
            self.layer_norm_2(x)
        )

        return x

# ============================================================
# TRANSFORMER LANGUAGE MODEL
# ============================================================


class TransformerLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_lookup_table = nn.Embedding(
            num_embeddings=max_token_value + 1,
            embedding_dim=config["d_model"]
        )

        self.transformer_blocks = nn.Sequential(
            *(
                [
                    TransformerBlock(
                        num_heads=config["num_heads"]
                    )

                    for _ in range(config["num_blocks"])
                ]

                + [nn.LayerNorm(config["d_model"])]
            )
        )

        self.language_model_out_linear_layer = nn.Linear(
            in_features=config["d_model"],
            out_features=max_token_value
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        position_encoding_lookup_table = torch.zeros(
            config["context_length"],
            config["d_model"]
        )

        position = torch.arange(
            0,
            config["context_length"],
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                config["d_model"],
                2
            ).float()

            * (-math.log(10000.0) / config["d_model"])
        )

        position_encoding_lookup_table[:, 0::2] = torch.sin(
            position * div_term
        )

        position_encoding_lookup_table[:, 1::2] = torch.cos(
            position * div_term
        )

        position_embedding = position_encoding_lookup_table[:T, :].to(device)

        x = self.token_embedding_lookup_table(idx) + position_embedding

        x = self.transformer_blocks(x)

        logits = self.language_model_out_linear_layer(x)

        if targets is not None:

            B, T, C = logits.shape

            logits_reshaped = logits.view(B * T, C)

            targets_reshaped = targets.view(B * T)

            loss = F.cross_entropy(
                input=logits_reshaped,
                target=targets_reshaped
            )

        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_crop = idx[:, -config["context_length"]:]

            logits, loss = self(idx_crop)

            logits_last_timestep = logits[:, -1, :]

            probs = F.softmax(
                input=logits_last_timestep,
                dim=-1
            )

            idx_next = torch.multinomial(
                input=probs,
                num_samples=1
            )

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# ============================================================
# INITIALIZE MODEL
# ============================================================

model = TransformerLanguageModel().to(device)

# ============================================================
# GET BATCH
# ============================================================


def get_batch(split: str):

    data = train_data if split == "train" else val_data

    idxs = torch.randint(
        low=0,
        high=len(data) - config["context_length"],
        size=(config["batch_size"],)
    )

    x = torch.stack(
        [
            data[idx:idx + config["context_length"]]
            for idx in idxs
        ]
    ).to(device)

    y = torch.stack(
        [
            data[idx + 1:idx + config["context_length"] + 1]
            for idx in idxs
        ]
    ).to(device)

    return x, y

# ============================================================
# ESTIMATE LOSS
# ============================================================


@torch.no_grad()
def estimate_loss():

    out = {}

    model.eval()

    for split in ["train", "valid"]:

        losses = torch.zeros(config["eval_iters"])

        for k in range(config["eval_iters"]):

            x_batch, y_batch = get_batch(split)

            logits, loss = model(x_batch, y_batch)

            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=config["learning_rate"]
)

tracked_losses = []

# ============================================================
# TRAINING LOOP
# ============================================================
if not os.path.exists('model-ckpt.pt'):
    for step in range(config["max_iters"]):

        if (
            step % config["eval_interval"] == 0
            or step == config["max_iters"] - 1
        ):

            losses = estimate_loss()

            tracked_losses.append(losses)

            print(
                "Step:",
                step,
                "Training Loss:",
                round(losses["train"].item(), 3),
                "Validation Loss:",
                round(losses["valid"].item(), 3)
            )

        xb, yb = get_batch("train")

        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

    torch.save(model.state_dict(), "model-ckpt.pt")
else:
    model.load_state_dict(torch.load("model-ckpt.pt", map_location=device))

# ============================================================
# TEST MODEL
# ============================================================

if not os.path.exists("test.txt"):
    url = "https://huggingface.co/datasets/Wannita/PyCoder/resolve/main/train.txt"

    with open("test.txt", "w") as f:
        f.write(requests.get(url).text)

with open("test.txt", "r", encoding="utf-8") as f:
    text = f.read()

N_CHARS = 300
prompt = text[:N_CHARS]

print("========== PROMPT ==========\n")
print(prompt)

# Encode prompt
input_ids = encoding.encode(prompt)

# Convert to tensor
x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

# Generate continuation
with torch.no_grad():
    generated_ids = model.generate(
        idx=x,
        max_new_tokens=200
    )

# Decode generated tokens
generated_text = encoding.decode(generated_ids[0].tolist())

print("\n========== GENERATED TEXT ==========\n")
print(generated_text)