import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import pretty_midi
import time
import matplotlib.pyplot as plt

from vae_model import VAE, loss_function

DATA_PATH = r"C:\Users\User\Pictures\CSE425\processed_data.npy"
OUTPUT_DIR = r"C:\Users\User\Pictures\CSE425\generated_midis"

BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
LATENT_DIM = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

start_time = time.time()

data = np.load(DATA_PATH)
data = data.reshape(-1, 128*128)

dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

epoch_losses = []

for epoch in range(EPOCHS):
    total_loss = 0

    for batch in loader:
        x = batch[0].to(device)

        x_hat, mu, logvar = model(x)
        loss = loss_function(x, x_hat, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "vae_model.pth")
print("Model saved ✅")

plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), epoch_losses, marker='o', linestyle='-', linewidth=2, markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('VAE Training Loss Curve', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()
print("Loss curve saved as 'loss_curve.png' ✅")

def piano_roll_to_midi(piano_roll, filename):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    for pitch in range(128):
        notes = piano_roll[:, pitch]
        start = None

        for t in range(len(notes)):
            if notes[t] > 0 and start is None:
                start = t
            elif notes[t] == 0 and start is not None:
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=start/16,
                    end=t/16
                )
                instrument.notes.append(note)
                start = None

    pm.instruments.append(instrument)
    pm.write(filename)

if os.path.exists(OUTPUT_DIR):
    for file in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, file))
else:
    os.makedirs(OUTPUT_DIR)

model.eval()

for i in range(8):
    z = torch.randn(1, LATENT_DIM).to(device)

    with torch.no_grad():
        sample = model.decode(z).cpu().numpy()

    sample = sample.reshape(128, 128)
    sample = (sample > 0.3).astype(np.int32)

    piano_roll_to_midi(sample, os.path.join(OUTPUT_DIR, f"sample_{i}.mid"))

print("Generated 8 MIDI files ✅")

z1 = torch.randn(1, LATENT_DIM).to(device)
z2 = torch.randn(1, LATENT_DIM).to(device)

alphas = np.linspace(0, 1, 5)

for i, alpha in enumerate(alphas):
    z = (1 - alpha) * z1 + alpha * z2

    with torch.no_grad():
        sample = model.decode(z).cpu().numpy()

    sample = sample.reshape(128, 128)
    sample = (sample > 0.3).astype(np.int32)

    piano_roll_to_midi(sample, os.path.join(OUTPUT_DIR, f"interp_{i}.mid"))

print("Interpolation done ✅")

pitch_usage = data.sum(axis=0)
active_pitches = np.count_nonzero(pitch_usage)

print("Active pitches used:", active_pitches)

end_time = time.time()
print(f"\nTotal Execution Time: {(end_time - start_time)/60:.2f} minutes")