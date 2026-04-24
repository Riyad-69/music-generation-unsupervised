import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import pretty_midi
from transformer_model import MusicTransformer

# ======================
# CONFIG
# ======================
DATA_PATH = r"C:\Users\User\Pictures\CSE425\token_data.npy"
OUTPUT_DIR = r"C:\Users\User\Pictures\CSE425\transformer_midis"

BATCH_SIZE = 64
EPOCHS = 20   # 🔥 increased
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# LOAD DATA
# ======================
data = np.load(DATA_PATH)

inputs = data[:, :-1]
targets = data[:, 1:]

dataset = TensorDataset(
    torch.tensor(inputs, dtype=torch.long),
    torch.tensor(targets, dtype=torch.long)
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ======================
# MODEL
# ======================
model = MusicTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# ======================
# TRAIN
# ======================
for epoch in range(EPOCHS):
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        output = model(x)

        loss = criterion(output.reshape(-1, 129), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")

# ======================
# PERPLEXITY
# ======================
avg_loss = total_loss / len(loader)
perplexity = np.exp(avg_loss)
print("Perplexity:", perplexity)

# ======================
# MIDI FUNCTION
# ======================
def tokens_to_midi(tokens, filename):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    time = 0
    for t in tokens:
        if t > 0:
            note = pretty_midi.Note(
                velocity=100,
                pitch=int(t-1),
                start=time,
                end=time+0.25
            )
            instrument.notes.append(note)
        time += 0.25

    pm.instruments.append(instrument)
    pm.write(filename)

# ======================
# GENERATION (IMPROVED 🔥)
# ======================
os.makedirs(OUTPUT_DIR, exist_ok=True)

model.eval()

temperature = 1.2   # 🔥 randomness control

for i in range(10):
    seq = [0]*20

    for _ in range(300):

        x = torch.tensor(seq[-128:], dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)

        probs = torch.softmax(out[0, -1] / temperature, dim=0).cpu().numpy()

        next_token = np.random.choice(len(probs), p=probs)

        # 🔥 reduce repetition
        if len(seq) > 5 and next_token == seq[-1]:
            next_token = np.random.randint(1, 128)

        if next_token == 0:
            next_token = np.random.randint(1, 128)

        seq.append(next_token)

    tokens_to_midi(seq, os.path.join(OUTPUT_DIR, f"song_{i}.mid"))

print("Generated 10 long sequences ✅")