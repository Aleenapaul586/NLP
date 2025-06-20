from textblob import TextBlob

# Input text
text = "I love learning Python. It is very interesting and fun!"

# Create a TextBlob object
blob = TextBlob(text)

# Sentence Tokenization
print("Sentences:")
for sentence in blob.sentences:
    print("-", sentence)

# Word Tokenization
print("\nWords:")
for word in blob.words:
    print("-", word)

# POS Tagging
print("\nPart-of-Speech Tags:")
print(blob.tags)

# Sentiment Analysis
print("\nSentiment Analysis:")
print("Polarity:", blob.sentiment.polarity)
print("Subjectivity:", blob.sentiment.subjectivity)
