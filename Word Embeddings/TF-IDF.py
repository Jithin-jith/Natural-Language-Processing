#Term Frequency-Inverse Document Frequency

"""TF-IDF (Term Frequency-Inverse Document Frequency) is an advanced version of Bag of Words (BoW) 
that assigns weights to words based on their importance in a document relative to the entire corpus."""

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('english')

# Sample documents
documents = [
    "I love programming in Python.",
    "Python is great for machine learning.",
    "I enjoy learning new programming languages."
]

# Step 2: Create a TfidfVectorizer instance
vectorizer = TfidfVectorizer(stop_words=stop_words)  # Includes single-character words like 'I'

# Step 3: Fit and transform the documents
X = vectorizer.fit_transform(documents)

# Step 4: Convert to an array and print feature names and matrix
print("Feature Names:", vectorizer.get_feature_names_out())
print("Shape of the corpus array vector : ",X.toarray().shape)
print("TF-IDF Representation:\n", X.toarray())
