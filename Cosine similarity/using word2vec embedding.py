import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained Word2Vec model (Google News)
word2vec_model = api.load("word2vec-google-news-300")

# Define words
word1 = "king"
word2 = "queen"
word3 = "apple"

# Get word vectors
vec1 = word2vec_model[word1]
vec2 = word2vec_model[word2]
vec3 = word2vec_model[word3]

# Compute cosine similarity
similarity_1_2 = cosine_similarity([vec1], [vec2])[0][0]  # king vs queen
similarity_1_3 = cosine_similarity([vec1], [vec3])[0][0]  # king vs apple

# Print results
print(f"Cosine Similarity between '{word1}' and '{word2}': {similarity_1_2:.4f}")
print(f"Cosine Similarity between '{word1}' and '{word3}': {similarity_1_3:.4f}")
