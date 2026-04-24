import numpy as np

def pitch_histogram(seq):
    hist = np.zeros(12)

    for note in seq:
        if note > 0:
            hist[note % 12] += 1

    return hist / hist.sum()
def rhythm_diversity(seq):
    durations = []
    count = 1

    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            count += 1
        else:
            durations.append(count)
            count = 1

    return len(set(durations)) / len(durations)
def repetition_ratio(seq):
    repeats = sum(1 for i in range(1,len(seq)) if seq[i]==seq[i-1])
    return repeats / len(seq)