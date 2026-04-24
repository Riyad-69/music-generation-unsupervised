import numpy as np
import matplotlib.pyplot as plt
from metrics import *

data = np.load("token_data.npy")

# take multiple sequences
all_notes = data[:100].flatten()

hist = np.zeros(12)

for note in all_notes:
    if note > 0:
        hist[note % 12] += 1

hist = hist / hist.sum()
plt.bar(range(12), hist)
plt.title("Pitch Distribution (Improved)")
plt.xlabel("Pitch Class")
plt.ylabel("Frequency")
plt.show()