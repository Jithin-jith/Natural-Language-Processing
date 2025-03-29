#Word Embeddings with Word2Vec

"""Unlike BoW and TF-IDF, which treat words as independent entities, 
Word2Vec represents words as dense vectors in a continuous vector space, capturing semantic relationships."""

#Word2Vec has two main training approaches:
"""
1. Continuous Bag of Words (CBOW)
2. Skip-Gram
"""

import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Sample documents
documents = [
    "I love programming in Python.",
    "Python is great for machine learning.",
    "I enjoy learning new programming languages."
]

# Step 2: Tokenize the text
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]

# Train CBOW Model

# CBOW model (sg=0 for CBOW)
cbow_model = Word2Vec(sentences=tokenized_docs, vector_size=50, window=3, min_count=1, workers=4, sg=0)

# Print word vector for 'python'
print("CBOW Vector for 'python':\n", cbow_model.wv['python'])

# Find similar words to 'learning'
print("\nCBOW Similar words to 'learning':", cbow_model.wv.most_similar('learning'))

# Train Skip-Gram Model

# Skip-Gram model (sg=1 for Skip-Gram)
skipgram_model = Word2Vec(sentences=tokenized_docs, vector_size=50, window=3, min_count=1, workers=4, sg=1)

# Print word vector for 'python'
print("\nSkip-Gram Vector for 'python':\n", skipgram_model.wv['python'])

# Find similar words to 'learning'
print("\nSkip-Gram Similar words to 'learning':", skipgram_model.wv.most_similar('learning'))

    