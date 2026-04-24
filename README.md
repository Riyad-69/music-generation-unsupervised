#  Unsupervised Music Generation using Deep Learning

This project implements multiple deep learning models for **multi-genre music generation** using MIDI data.
It explores unsupervised learning techniques including Autoencoder, Variational Autoencoder (VAE), Transformer, and Reinforcement Learning from Human Feedback (RLHF).

---

#  Features

*  MIDI preprocessing pipeline (piano-roll + tokenization)
*  Multiple models:

  * LSTM Autoencoder (Task 1)
  * Variational Autoencoder (Task 2)
  * Transformer-based Generator (Task 3)
  * RLHF Fine-tuning (Task 4)
*  Evaluation metrics (rhythm, repetition, pitch distribution)
*  MIDI generation outputs
*  Visualization plots

---

#  Installation

## 1. Clone the repository

```
git clone <repository-url>
cd 22101201_CSE425_Project
```

## 2. Create virtual environment (recommended)

```
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux
```

## 3. Install dependencies

```
pip install -r requirements.txt
```

---

#  Dataset Setup

You need MIDI datasets:

* Groove MIDI Dataset
* Lakh MIDI Dataset (clean subset)

Place MIDI files in the `clean_midi/` directory, then run preprocessing:

```
python prepocessing.py
```

This generates the required files:

* `processed_data.npy` - Processed piano roll data for VAE
* `token_data.npy` - Tokenized sequences for Transformer

---

#  How to Run

##  Step 1: Preprocess Data

```
python prepocessing.py
```

✔ Output:

* `processed_data.npy` - Piano roll format
* `token_data.npy` - Tokenized sequences

---

##  Step 2: Train VAE (Task 2)

```
python train_vae.py
```

 Output:

* Trained model
* Generated MIDI samples

---

##  Step 3: Train Transformer (Task 3)

```
python train_transformer.py
```

✔ Output:

* Perplexity score
* Generated long music sequences

---

##  Step 4: Run RLHF (Task 4)

```
python task4_rlhf.py
```

 Output:

* Improved MIDI samples
* Reward-based optimization

---

##  Step 5: Run Baselines

```
python baselines.py
```

✔ Output:

* Random generator MIDI
* Markov chain MIDI

---

##  Step 6: Evaluation

```
python metrics.py
```

✔ Output:

* Pitch distribution
* Rhythm diversity
* Repetition ratio

---

#  Outputs

Generated files include:

* MIDI compositions
* Evaluation plots

---

#  Models Overview

| Model       | Description                           |
| ----------- | ------------------------------------- |
| Autoencoder | Learns reconstruction of sequences    |
| VAE         | Learns probabilistic latent space     |
| Transformer | Autoregressive sequence generation    |
| RLHF        | Improves output using reward function |

---

#  Metrics Used

* Pitch Histogram Similarity
* Rhythm Diversity
* Repetition Ratio
* Human Evaluation Score

---

# GPU Support

If you have NVIDIA GPU:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check:

```
python -c "import torch; print(torch.cuda.is_available())"
```

---

#  Notes

* Transformer training benefits significantly from GPU
* MIDI outputs use piano instrument by default
* Results may vary depending on dataset

---

#  Report

The final report includes:

* Methodology
* Results and comparison
* Evaluation metrics

---

#  Author

MD READ AL RASID
ID : 22101201
CSE425 / Neural Networks Project

---

#  Acknowledgements

* Groove MIDI Dataset
* Lakh MIDI Dataset
* PyTorch
* pretty_midi



