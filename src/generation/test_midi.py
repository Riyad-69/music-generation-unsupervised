import pretty_midi
import os

midi_path = r"C:\Users\User\Pictures\CSE425\clean_midi"

# Recursively find MIDI files
midi_files = []

for root, dirs, files in os.walk(midi_path):
    for file in files:
        if file.endswith(".mid") or file.endswith(".midi"):
            midi_files.append(os.path.join(root, file))

print("Total MIDI files found:", len(midi_files))

# Take first file
first_file = midi_files[0]

print("Loading:", first_file)

# Load MIDI
midi_data = pretty_midi.PrettyMIDI(first_file)

# Print info
print("Number of instruments:", len(midi_data.instruments))

for instrument in midi_data.instruments:
    print("Instrument:", instrument.program, "Notes:", len(instrument.notes))