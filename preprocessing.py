# ================================
# Text Preprocessing Example
# Libraries: NLTK, spaCy
# Steps: Tokenization, Stemming, Lemmatization, Stopword Removal
# ================================

# Install required libraries if not already installed
# pip install nltk spacy
# python -m spacy download en_core_web_sm

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy

# Download necessary NLTK datasets (only first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# ------------------------
# Sample Text Corpus
# ------------------------
text_corpus = """The new AI technology is transforming industries worldwide. 
Many companies are adopting artificial intelligence to improve efficiency and productivity."""

print("Original Text:")
print(text_corpus)
print("="*60)

# ------------------------
# 1. TOKENIZATION
# ------------------------
tokens = word_tokenize(text_corpus)
print("\n1. Tokenization:")
print(tokens)

# ------------------------
# 2. STOP WORD REMOVAL
# ------------------------
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w.lower() not in stop_words and w.isalpha()]
print("\n2. Stop Word Removal:")
print(filtered_tokens)

# ------------------------
# 3. STEMMING (using Porter Stemmer)
# ------------------------
ps = PorterStemmer()
stemmed_tokens = [ps.stem(w) for w in filtered_tokens]
print("\n3. Stemming:")
print(stemmed_tokens)

# ------------------------
# 4. LEMMATIZATION (using spaCy)
# ------------------------
doc = nlp(" ".join(filtered_tokens))
lemmatized_tokens = [token.lemma_ for token in doc]
print("\n4. Lemmatization:")
print(lemmatized_tokens)
