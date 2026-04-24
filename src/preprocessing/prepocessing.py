import pretty_midi
import os
import numpy as np
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
DATA_PATH = r"C:\Users\User\Pictures\CSE425\clean_midi"
SAVE_PATH = r"C:\Users\User\Pictures\CSE425\processed_data.npy"

FS = 16                 # timing resolution (16 steps per second)
SEQ_LEN = 128           # sequence length (time steps)
MAX_FILES = 2000        # limit for testing (increase later)

# ==============================
# STEP 1: COLLECT MIDI FILES
# ==============================
midi_files = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".mid") or file.endswith(".midi"):
            midi_files.append(os.path.join(root, file))

print("Total MIDI files found:", len(midi_files))

# Optional: limit dataset (for faster testing)
midi_files = midi_files[:MAX_FILES]

print("Using files:", len(midi_files))

# ==============================
# STEP 2: PROCESS FUNCTION
# ==============================
def process_midi(file_path):
    try:
        midi = pretty_midi.PrettyMIDI(file_path)

        # Convert to piano roll
        piano_roll = midi.get_piano_roll(fs=FS)

        # Transpose → (time_steps, 128)
        piano_roll = piano_roll.T

        # Normalize → binary
        piano_roll = (piano_roll > 0).astype(np.float32)

        return piano_roll

    except Exception as e:
        return None

# ==============================
# STEP 3: SEGMENT FUNCTION
# ==============================
def segment_sequence(piano_roll, seq_len):
    sequences = []

    total_steps = piano_roll.shape[0]

    for i in range(0, total_steps - seq_len, seq_len):
        seq = piano_roll[i:i + seq_len]
        sequences.append(seq)

    return sequences

# ==============================
# STEP 4: MAIN LOOP
# ==============================
all_sequences = []

for file in tqdm(midi_files):
    piano_roll = process_midi(file)

    if piano_roll is None:
        continue

    sequences = segment_sequence(piano_roll, SEQ_LEN)

    all_sequences.extend(sequences)

# ==============================
# STEP 5: CONVERT TO NUMPY
# ==============================
dataset = np.array(all_sequences)

print("\nFinal dataset shape:", dataset.shape)

# Expected:
# (num_sequences, 128, 128)

# ==============================
# STEP 6: SAVE DATASET
# ==============================
np.save(SAVE_PATH, dataset)

print("\nDataset saved at:", SAVE_PATH)