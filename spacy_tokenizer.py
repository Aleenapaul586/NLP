import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Input text
text = "Kochi, The Kerala High Court on Friday quashed a case booked under the POCSO Act against six journalists of Malayalam news channel Asianet for allegedly disclosing the identity of a minor victim of sexual assault, in a programme on the ill effects of drug abuse two years ago."

# Process the text
doc = nlp(text)

# Print each token
print("Tokens:")
for token in doc:
    print(token.text)
