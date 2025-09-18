# Import libraries
import pandas as pd
import re
import html
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Load the dataset
df = pd.read_csv("tweets_sample.csv")  # replace with your file path

# Check the column names
print("Columns in dataset:", df.columns)

# Use the correct column for tweets
texts = df['Tweet Content'].dropna().astype(str)

# Function to clean tweets
def clean_tweet(text):
    text = text.lower()  # lowercase
    text = html.unescape(text)  # decode HTML codes like &amp;
    text = re.sub(r'http\S+', '', text)  # remove URLs
    text = re.sub(r'@\w+', '', text)  # remove mentions
    text = re.sub(r'#\w+', '', text)  # remove hashtags
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# Apply cleaning
cleaned_texts = texts.apply(clean_tweet)

print("First 5 cleaned tweets:\n", cleaned_texts.head(), "\n")

# ------------------------------
# a. Bag-of-Words (BoW) Representation
# ------------------------------
bow_vectorizer = CountVectorizer(stop_words='english')
bow_matrix = bow_vectorizer.fit_transform(cleaned_texts)

print("Bag-of-Words matrix shape:", bow_matrix.shape)
print("First 5 feature vectors:\n", bow_matrix.toarray()[:5])
print("Feature names (first 10):", bow_vectorizer.get_feature_names_out()[:10], "\n")

# ------------------------------
# b. TF-IDF Representation
# ------------------------------
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_texts)

print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("First 5 TF-IDF vectors:\n", tfidf_matrix.toarray()[:5])
print("TF-IDF Feature names (first 10):", tfidf_vectorizer.get_feature_names_out()[:10], "\n")
