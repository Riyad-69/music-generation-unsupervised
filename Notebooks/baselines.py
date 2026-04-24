import numpy as np
import pretty_midi
import os

OUTPUT_DIR = r"C:\Users\User\Pictures\CSE425\baseline_midis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def random_generator():
    seq = np.random.randint(1, 128, size=200)

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    time = 0
    for note in seq:
        instrument.notes.append(pretty_midi.Note(
            velocity=100,
            pitch=int(note),
            start=time,
            end=time+0.2
        ))
        time += 0.2

    pm.instruments.append(instrument)
    pm.write(os.path.join(OUTPUT_DIR, "random.mid"))

random_generator()
print("Random baseline done ✅")
from collections import defaultdict
import numpy as np
import pretty_midi
import os

DATA_PATH = r"C:\Users\User\Pictures\CSE425\token_data.npy"
OUTPUT_DIR = r"C:\Users\User\Pictures\CSE425\baseline_midis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# TRAIN MARKOV
# ======================
data = np.load(DATA_PATH)

transitions = defaultdict(list)

for seq in data[:500]:  # small subset (faster)
    for i in range(len(seq)-1):
        transitions[seq[i]].append(seq[i+1])

print("Markov trained ✅")

# ======================
# GENERATE SEQUENCE
# ======================
seq = [np.random.randint(1,128)]

for _ in range(200):
    prev = seq[-1]

    if prev in transitions:
        next_note = np.random.choice(transitions[prev])
    else:
        next_note = np.random.randint(1,128)

    seq.append(next_note)

# ======================
# SAVE MIDI
# ======================
pm = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)

time = 0
for note in seq:
    instrument.notes.append(pretty_midi.Note(
        velocity=100,
        pitch=int(note),
        start=time,
        end=time+0.2
    ))
    time += 0.2

pm.instruments.append(instrument)
pm.write(os.path.join(OUTPUT_DIR, "markov.mid"))

print("Markov MIDI generated ✅")