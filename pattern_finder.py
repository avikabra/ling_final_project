import spacy
import pandas as pd
from spacy.matcher import DependencyMatcher

# Load spaCy model and dataset
nlp = spacy.load('en_core_web_sm')
data = pd.read_csv("train.csv")
sentences = data['startphrase'].tolist()

# Function to filter sentences by pattern matching
def filter_sentences(sentences, patterns):
    matcher = DependencyMatcher(nlp.vocab)
    
    # Add patterns to the matcher
    for i, pattern in enumerate(patterns):
        matcher.add(f"pattern_{i+1}", [pattern])
    
    matched_sentences = []
    unmatched_sentences = []

    # Process each sentence
    for sentence in sentences:
        doc = nlp(sentence)
        matches = matcher(doc)
        if matches:
            matched_sentences.append(sentence)
        else:
            unmatched_sentences.append(sentence)
    
    return matched_sentences, unmatched_sentences

# Define initial patterns (expand these over iterations)
patterns = [
    # Pattern for "X had/have the Y of Z"
    [
        {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["have", "had"]}}},
        {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "object", "RIGHT_ATTRS": {"DEP": "dobj", "POS": "NOUN"}},
        {"LEFT_ID": "object", "REL_OP": ">", "RIGHT_ID": "comparison", "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": "of"}},
        {"LEFT_ID": "comparison", "REL_OP": ">", "RIGHT_ID": "vehicle", "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "prep"]}}},
    ],

    # Pattern for "X is as ADJ as Y"
    [
        {"RIGHT_ID": "copula", "RIGHT_ATTRS": {"LEMMA": "be"}}, # {"IN": ["is", "was"]}
        {"LEFT_ID": "copula", "REL_OP": ">", "RIGHT_ID": "adjective", "RIGHT_ATTRS": {"DEP": "acomp", "POS": "ADJ"}},
        {"LEFT_ID": "adjective", "REL_OP": ">", "RIGHT_ID": "comparison", "RIGHT_ATTRS": {"DEP": "prep", "LEMMA": "as"}},
        {"LEFT_ID": "comparison", "REL_OP": ">", "RIGHT_ID": "vehicle", "RIGHT_ATTRS": {"DEP": {"IN": ["pobj", "prep"]}}},
    ]


]
# New patterns based on the given sentences

# Iteratively match patterns and reduce the dataset
iteration = 1
while sentences:
    print(f"Iteration {iteration}: {len(sentences)} sentences remaining.")
    matched, unmatched = filter_sentences(sentences, patterns)
    print(f"Matched: {len(matched)}, Unmatched: {len(unmatched)}")

    # Add unmatched sentences for manual review
    if unmatched:
        print("Unmatched examples for inspection:")
        print(unmatched[:5])  # Display first few for manual pattern identification
    
    # Update sentence list to only include unmatched ones
    sentences = unmatched
    iteration += 1

    # Stop loop if no unmatched sentences remain
    if not unmatched:
        break

    break

# Add to the end of the existing program to process matched sentences
def process_matches(matched_sentences):
    print("in the function")
    rewritten_sentences = []
    for i, sentence in enumerate(matched_sentences):
        doc = nlp(sentence)
        vehicle = None
        
        # Extract vehicle from matched patterns
        for token in doc:
            if token.dep_ == "pobj":
                vehicle = token.text
                break
        
        # Rewrite the sentence with the desired ending if a vehicle is found
        if vehicle:
            rewritten_sentence = f"The metaphorical object is '{vehicle}'."
            rewritten_sentences.append(rewritten_sentence)

        # Print every 10th sentence to the terminal for validation
        if i % 10 == 0:
            print(sentence)
            print(rewritten_sentence)
    
    return rewritten_sentences

processed_results = process_matches(matched)
print(processed_results[:5])

# Learning from these patterns: Inspect subjects and their contexts in processed_results
# to identify new variations and refine/add new patterns iteratively.
