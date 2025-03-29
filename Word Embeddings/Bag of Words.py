"""When working with text data in Natural Language Processing (NLP), we need to convert textual data into 
numerical representations that machine learning models can understand. """

#One common techniques for this are Bag of Words (BoW) 

# The Bag of Words (BoW) model is a simple and effective way to convert text into numerical representations. 
# It represents text as a set of word frequencies, ignoring grammar and word order.

from sklearn.feature_extraction.text import CountVectorizer
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

# Step 2: Create a CountVectorizer instance
vectorizer = CountVectorizer(stop_words=stop_words)

# Step 3: Fit and transform the documents
X = vectorizer.fit_transform(documents)

# print(X)

# Step 4: Convert to an array and print feature names
print("Feature Names:", vectorizer.get_feature_names_out())
print("BoW Representation:\n", X.toarray())
