from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

# Download required tokenizer
nltk.download('punkt')

# Create a stemmer object
stemmer = PorterStemmer()

# Sample text
text = "Kochi, The Kerala High Court on Friday quashed a case booked under the POCSO Act against six journalists of Malayalam news channel Asianet for allegedly disclosing the identity of a minor victim of sexual assault, in a programme on the ill effects of drug abuse two years ago."

# Tokenize the text
words = word_tokenize(text)

# Stem each word
print("Stemming Result:")
for word in words:
    print(f"{word:15} → Stem: {stemmer.stem(word)}")
