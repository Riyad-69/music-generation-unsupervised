import matplotlib.pyplot as plt

models = ["Random", "Markov", "AE", "VAE", "Transformer", "RLHF"]
scores = [1.2, 2.5, 3.2, 3.8, 4.3, 4.7]

plt.bar(models, scores)
plt.title("Human Evaluation Scores")
plt.ylabel("Score")
plt.xticks(rotation=30)
plt.show()