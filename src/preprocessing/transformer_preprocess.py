import numpy as np

DATA_PATH = r"C:\Users\User\Pictures\CSE425\processed_data.npy"
SAVE_PATH = r"C:\Users\User\Pictures\CSE425\token_data.npy"

data = np.load(DATA_PATH)

token_sequences = []

for seq in data:
    tokens = []

    for t in seq:
        notes = np.where(t > 0)[0]

        if len(notes) == 0:
            tokens.append(0)  # silence
        else:
            tokens.append(notes[0] + 1)  # take first active note

    token_sequences.append(tokens)

token_sequences = np.array(token_sequences)

np.save(SAVE_PATH, token_sequences)

print("Tokenization done:", token_sequences.shape)