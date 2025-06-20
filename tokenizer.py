import nltk

nltk.download('punkt')  # Download tokenizer once

from nltk.tokenize import word_tokenize

text = "Kochi, The Kerala High Court on Friday quashed a case booked under the POCSO Act against six journalists of Malayalam news channel Asianet for allegedly disclosing the identity of a minor victim of sexual assault, in a programme on the ill effects of drug abuse two years ago."
tokens = word_tokenize(text)

print("Word Tokens:", tokens)
