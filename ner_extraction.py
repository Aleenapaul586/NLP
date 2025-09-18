# ner_extraction.py
"""
Named Entity Recognition (NER) on Job Postings & Scientific Articles
-------------------------------------------------------------------
This script uses spaCy's pre-trained NER model to extract
Persons and Organizations from unstructured text.

Steps:
1. Load dataset (job postings + scientific articles).
2. Apply spaCy NER.
3. Extract structured info (Person → Organization).
4. Save results to CSV.
"""

import spacy
import pandas as pd

# -------------------------------
# Step 1: Load spaCy Pre-trained Model
# -------------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------------
# Step 2: Dataset (Job Postings + Scientific Articles)
# -------------------------------
data = [
    # Job postings
    "We are looking for a Data Scientist at IBM Research in New York to work on AI and machine learning projects.",
    "Google is hiring a Software Engineer to join their Cloud Computing division in California.",
    "Amazon is seeking a Research Scientist in Seattle to advance natural language processing technologies.",
    "Microsoft is recruiting a Cybersecurity Analyst in Washington to strengthen cloud security solutions.",
    "Apple is hiring a Machine Learning Engineer in Cupertino to work on computer vision projects.",
    
    # Scientific articles
    "Dr. Emily Carter from Harvard University published an article on renewable energy systems in Nature Journal.",
    "Prof. James Wilson at MIT collaborated with Microsoft on a paper about quantum computing.",
    "A recent study by Dr. Susan Lee from Stanford University explored AI applications in healthcare.",
    "Dr. Robert Brown from Oxford University co-authored a publication with researchers at Google DeepMind.",
    "Prof. Michael Green at Cambridge University presented his findings on blockchain in IEEE Transactions."
]

# -------------------------------
# Step 3: Perform NER and Extract Info
# -------------------------------
records = []

for text in data:
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

    # Save combinations Person → Organization
    for person in persons:
        for org in orgs:
            records.append({"Person": person, "Organization": org, "Sentence": text})

# -------------------------------
# Step 4: Save Results as Table
# -------------------------------
df = pd.DataFrame(records)

print("\nExtracted Entities:\n")
print(df)

# Save to CSV
df.to_csv("ner_results.csv", index=False)
print("\nResults saved to ner_results.csv")
