import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# Set a random seed
random_seed = 42
random.seed(random_seed)

# Load two Sentence-BERT models
model1 = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")  # Erstes Modell
model2 = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")  # Zweites Modell, hier als Beispiel

# Function to read sentence pairs from a text file
def read_sentence_pairs(file_path):
    sentence_pairs = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentence1, sentence2 = line.strip().split('|')
            sentence_pairs.append((sentence1, sentence2))
    return sentence_pairs

# Read the sentence pairs from the file
sentence_pairs = read_sentence_pairs('3.5v3.5NEU.txt')

# Initialize lists to store similarity scores
similarity_scores_model1 = []
similarity_scores_model2 = []

# Iterate over each pair of sentences
for sentence1, sentence2 in sentence_pairs:
    # Compute embeddings for the sentences with the first model
    embeddings_model1 = model1.encode([sentence1, sentence2])
    # Compute cosine similarity between the sentence embeddings for the first model
    similarity_score_model1 = cosine_similarity([embeddings_model1[0]], [embeddings_model1[1]])
    similarity_scores_model1.append(similarity_score_model1[0][0])

    # Compute embeddings for the sentences with the second model
    embeddings_model2 = model2.encode([sentence1, sentence2])
    # Compute cosine similarity between the sentence embeddings for the second model
    similarity_score_model2 = cosine_similarity([embeddings_model2[0]], [embeddings_model2[1]])
    similarity_scores_model2.append(similarity_score_model2[0][0])

# Convert the similarity scores to NumPy arrays for easy manipulation
similarity_scores_model1 = np.array(similarity_scores_model1)
similarity_scores_model2 = np.array(similarity_scores_model2)

# Increase the number of bins for finer granularity
num_bins = 50

# Plot the distribution of cosine similarity scores with Matplotlib
plt.figure(figsize=(14, 6))

# Plot histogram for each model
plt.subplot(1, 2, 1)
plt.hist(similarity_scores_model1, bins=num_bins, range=(0, 1), color='blue', alpha=0.7, edgecolor='black')
plt.title('Verteilung der Kosinus-Ähnlichkeitswerte (Modell 1)')
plt.xlabel('Kosinus-Ähnlichkeitswert')
plt.ylabel('Häufigkeit')

plt.subplot(1, 2, 2)
plt.hist(similarity_scores_model2, bins=num_bins, range=(0, 1), color='green', alpha=0.7, edgecolor='black')
plt.title('Verteilung der Kosinus-Ähnlichkeitswerte (Modell 2)')
plt.xlabel('Kosinus-Ähnlichkeitswert')
plt.ylabel('Häufigkeit')

plt.tight_layout()
plt.show()

# Plot comparison of cosine similarity scores for both models on an index-based line graph
plt.figure(figsize=(12, 6))

# Plot the values for each model with line connection
index_values = range(len(similarity_scores_model1))
plt.plot(index_values, similarity_scores_model1, marker='o', color='blue', label="Model 1 - all-MiniLM-L12-v2", linestyle='-')
plt.plot(index_values, similarity_scores_model2, marker='x', color='green', label="Model 2 - paraphrase-MiniLM-L6-v2", linestyle='--')

# Add labels and title
plt.title("Cosine Similarity Comparison by Index")
plt.xlabel("Index of Sentence Pair")
plt.ylabel("Cosine Similarity Score")
plt.legend()
plt.show()

print("Cosine Similarity Scores (Model 1):", similarity_scores_model1)
print("Cosine Similarity Scores (Model 2):", similarity_scores_model2)
