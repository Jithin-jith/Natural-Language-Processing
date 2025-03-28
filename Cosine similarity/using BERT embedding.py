from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model (SentenceTransformer)
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Define words
words = ["king", "queen", "apple"]

# Get BERT embeddings
embeddings = model.encode(words)

# Compute cosine similarity
similarity_matrix = cosine_similarity(embeddings)

# Print similarity results
for i in range(len(words)):
    for j in range(i + 1, len(words)):
        print(f"Cosine Similarity between '{words[i]}' and '{words[j]}': {similarity_matrix[i][j]:.4f}")
