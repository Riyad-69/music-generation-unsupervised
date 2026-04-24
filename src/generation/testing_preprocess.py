import numpy as np

data = np.load(r"C:\Users\User\Pictures\CSE425\processed_data.npy")

print("Shape:", data.shape)
print("Min:", data.min())
print("Max:", data.max())