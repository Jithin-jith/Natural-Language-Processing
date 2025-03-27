import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree

# Download necessary resources (only needed once)
# nltk.download()
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

# Load Spacy English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Apple is looking at buying U.K. startup for $1 billion. Elon Musk, the CEO of Tesla, visited Berlin."

# 1. Tokenization
print("\n--- Tokenization ---")
sentences = sent_tokenize(text)
print("Sentences:", sentences)
words = word_tokenize(text)
print("Words:", words)

# 2. Lemmatization & Stemming
print("\n--- Lemmatization & Stemming ---")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stemmed_words = [stemmer.stem(word) for word in words]
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print("Stemmed Words:", stemmed_words)
print("Lemmatized Words:", lemmatized_words)

# 3. Stopword Removal
print("\n--- Stopword Removal ---")
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print("Filtered Words:", filtered_words)

# 4. Part-of-Speech (POS) Tagging
print("\n--- Part-of-Speech (POS) Tagging ---")
pos_tags = pos_tag(words)
print("POS Tags:", pos_tags)

# 5. Named Entity Recognition (NER)
print("\n--- Named Entity Recognition (NER) ---")
# Using NLTK
nltk_ne_tree = ne_chunk(pos_tags)
named_entities = []
for subtree in nltk_ne_tree:
    if isinstance(subtree, Tree):  # If it's a named entity chunk
        entity_name = " ".join([token for token, pos in subtree.leaves()])
        entity_type = subtree.label()
        named_entities.append((entity_name, entity_type))
print("Named Entities (NLTK):", named_entities)

# Using spaCy
print("\nNamed Entities (spaCy):")
doc = nlp(text)
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")