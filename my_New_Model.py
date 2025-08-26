import requests
from bs4 import BeautifulSoup
import numpy as np

# === 1. SCRAPE WEBSITE ===
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return ' '.join(soup.get_text().split())

# === 2. TOKENIZATION ===
def build_vocab(text):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def tokenize(text, stoi):
    return [stoi[c] for c in text]

def detokenize(tokens, itos):
    return ''.join(itos[i] for i in tokens)

# === 3. MODEL (Simple RNN-like) ===
class TinyCharModel:
    def __init__(self, vocab_size, hidden_size=128, seq_len=32, lr=0.01):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lr = lr

        # Parameters (random init)
        self.Wxh = np.random.randn(vocab_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(hidden_size, vocab_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, vocab_size))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, inputs, h_prev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = h_prev
        for t in range(len(inputs)):
            xs[t] = np.zeros((1, self.vocab_size))
            xs[t][0][inputs[t]] = 1
            hs[t] = np.tanh(xs[t] @ self.Wxh + hs[t - 1] @ self.Whh + self.bh)
            ys[t] = hs[t] @ self.Why + self.by
            ps[t] = self.softmax(ys[t])
        return xs, hs, ys, ps

    def loss_and_gradients(self, inputs, targets, h_prev):
        xs, hs, ys, ps = self.forward(inputs, h_prev)
        loss = 0
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])

        for t in reversed(range(len(inputs))):
            loss += -np.log(ps[t][0][targets[t]])
            dy = np.copy(ps[t])
            dy[0][targets[t]] -= 1
            dWhy += hs[t].T @ dy
            dby += dy
            dh = dy @ self.Why.T + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh
            dbh += dh_raw
            dWxh += xs[t].T @ dh_raw
            dWhh += hs[t - 1].T @ dh_raw
            dh_next = dh_raw @ self.Whh.T

        # Clip gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        # Update weights
        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.Why -= self.lr * dWhy
        self.bh -= self.lr * dbh
        self.by -= self.lr * dby

        return loss, hs[len(inputs) - 1]

    def generate(self, seed_idx, length, h=None):
        x = np.zeros((1, self.vocab_size))
        x[0][seed_idx] = 1
        if h is None:
            h = np.zeros((1, self.hidden_size))
        output = [seed_idx]

        for _ in range(length):
            h = np.tanh(x @ self.Wxh + h @ self.Whh + self.bh)
            y = h @ self.Why + self.by
            p = self.softmax(y)
            idx = np.random.choice(range(self.vocab_size), p=p.ravel())
            output.append(idx)
            x = np.zeros((1, self.vocab_size))
            x[0][idx] = 1
        return output

# === 4. RUN EVERYTHING ===
if __name__ == "__main__":
    url = "https://finance.yahoo.com"
    text = scrape_website(url)
    text = text[:10000]  # Limit for demo
    stoi, itos = build_vocab(text)
    encoded = tokenize(text, stoi)

    model = TinyCharModel(vocab_size=len(stoi), hidden_size=128, seq_len=32, lr=0.01)

    # Training
    h_prev = np.zeros((1, model.hidden_size))
    for epoch in range(1000):
        if len(encoded) > model.seq_len + 1:
            idx = np.random.randint(0, len(encoded) - model.seq_len - 1)
        else:
            raise ValueError("Encoded data is too short for the specified sequence length.")
        #idx = np.random.randint(0, len(encoded) - model.seq_len - 1)
        input_seq = encoded[idx:idx + model.seq_len]
        target_seq = encoded[idx + 1:idx + model.seq_len + 1]
        loss, h_prev = model.loss_and_gradients(input_seq, target_seq, h_prev)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            sample = model.generate(seed_idx=input_seq[0], length=200)
            print(detokenize(sample, itos))
            print("=" * 50)