import torch
import numpy as np
import os
import pretty_midi
from transformer_model import MusicTransformer

# ======================
# CONFIG
# ======================
OUTPUT_DIR = r"C:\Users\User\Pictures\CSE425\rlhf_midis"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 1e-5   # small learning rate for RL
RL_STEPS = 5

# ======================
# LOAD MODEL
# ======================
model = MusicTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
# REWARD FUNCTION (SIMULATED HUMAN)
# ======================
def reward_function(seq):
    unique_notes = len(set(seq))
    repetition = sum(1 for i in range(1, len(seq)) if seq[i] == seq[i-1])
    return unique_notes - 0.5 * repetition

# ======================
# GENERATE SEQUENCE + LOG PROB
# ======================
def generate_sequence(model):
    seq = [0]*20
    log_probs = []

    for _ in range(200):
        x = torch.tensor(seq[-128:], dtype=torch.long).unsqueeze(0).to(device)

        out = model(x)
        probs = torch.softmax(out[0, -1], dim=0)

        dist = torch.distributions.Categorical(probs)
        next_token = dist.sample()

        log_probs.append(dist.log_prob(next_token))

        token = next_token.item()
        if token == 0:
            token = np.random.randint(1, 128)

        seq.append(token)

    return seq, torch.stack(log_probs)

# ======================
# BEFORE RL (BASELINE)
# ======================
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Generating BEFORE RL samples...")
before_samples = []

for i in range(5):
    seq, _ = generate_sequence(model)
    tokens_to_midi(seq, os.path.join(OUTPUT_DIR, f"before_{i}.mid"))
    before_samples.append(seq)

# ======================
# RL TRAINING LOOP
# ======================
print("Starting RLHF training...")

for step in range(RL_STEPS):
    optimizer.zero_grad()

    seq, log_probs = generate_sequence(model)
    reward = reward_function(seq)

    loss = -reward * log_probs.mean()  # policy gradient

    loss.backward()
    optimizer.step()

    print(f"Step {step+1}, Reward: {reward:.2f}")

# ======================
# AFTER RL
# ======================
print("Generating AFTER RL samples...")

for i in range(10):
    seq, _ = generate_sequence(model)
    tokens_to_midi(seq, os.path.join(OUTPUT_DIR, f"after_{i}.mid"))

print("RLHF completed ✅")