import spacy
import pandas as pd
from spacy.matcher import DependencyMatcher

nlp = spacy.load('en_core_web_sm')

def apply_template(start_phrase, ending1, ending2, start_pattern, end_patterns, subj_idx, simile_idx, end_subj_idx):
    start_doc = nlp(start_phrase)
    start_matcher = DependencyMatcher(nlp.vocab)
    start_matcher.add("start", [start_pattern])
    start_matches = start_matcher(start_doc)
    if len(start_matches) != 1:
        # print(len(start_matches), start_phrase)
        return None, None, None

    indices = start_matches[0][1]
    subject = start_doc[indices[subj_idx]].lemma
    simile_np = [token for token in start_doc[indices[simile_idx]].subtree]
    # print(subject, [item.text for item in simile_np])

    end_doc1, end_doc2 = nlp(ending1), nlp(ending2)
    end_matcher = DependencyMatcher(nlp.vocab)
    end_matcher.add("end", end_patterns)
    end_matches1, end_matches2 = end_matcher(end_doc1), end_matcher(end_doc2)
    if len(end_matches1) != 1 or len(end_matches2) != 1:
        # print("Ending does not match")
        # if len(end_matches1) == 1: print("Ending 1 matches")
        # else: print("doc1:", [token.pos_ for token in end_doc1])
        # if len(end_matches2) == 1: print("Ending 2 matches")
        # else: print("doc2:", [(token.pos_, token.dep_) for token in end_doc2])
        return None, None, None
    
    indices1, indices2 = end_matches1[0][1], end_matches2[0][1]
    end_subj1, end_subj2 = end_doc1[indices1[end_subj_idx]], end_doc2[indices2[end_subj_idx]]
    if (end_subj1.lemma != subject and end_subj1.lemma_ != "it" and end_subj1.pos_ != "PRON") or (end_subj2.lemma != subject and end_subj2.lemma_ != "it" and end_subj2.pos_ != "PRON"):
        # print(start_doc[indices[subj_idx]].lemma_, end_subj1.lemma_)
        return None, None, None
    
    np1, np2 = [token for token in end_subj1.subtree], [token for token in end_subj2.subtree]
    
    mod1 = end_doc1[:np1[0].i].text_with_ws
    for token in simile_np: mod1 += token.text_with_ws
    mod1 = mod1.rstrip() + end_doc1[np1[-1].i].whitespace_ + end_doc1[np1[-1].i+1:].text_with_ws

    mod2 = end_doc2[:np2[0].i].text_with_ws
    for token in simile_np: mod2 += token.text_with_ws
    mod2 = mod2.rstrip() + end_doc2[np2[-1].i].whitespace_ + end_doc2[np2[-1].i+1:].text_with_ws

    # still need to fix conjugation

    return "".join([token.text_with_ws for token in simile_np]), mod1.capitalize(), mod2.capitalize()

# start_matcher = DependencyMatcher(nlp.vocab)
patterns = []
subj_indices = []
simile_indices = []
patterns.append([
    {
        "RIGHT_ID": "principal_verb",
        "RIGHT_ATTRS": {"POS": "VERB", "DEP": "ROOT", "LEMMA": "have"}
    },
    {
        "LEFT_ID": "principal_verb",
        "REL_OP": ">",
        "RIGHT_ID": "object",
        "RIGHT_ATTRS": {"DEP": "dobj"}
    },
    {
        "LEFT_ID": "object",
        "REL_OP": ">",
        "RIGHT_ID": "prep",
        "RIGHT_ATTRS": {"text": "of"}
    },
    {
        "LEFT_ID": "prep",
        "REL_OP": ">",
        "RIGHT_ID": "simile_object",
        "RIGHT_ATTRS": {"DEP": "pobj"}
    },
    {
        "LEFT_ID": "principal_verb",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"DEP": {"IN":["nsubj", "csubj"]}}
    }
])
subj_indices.append(4)
simile_indices.append(3)

patterns.append([
    {
        "RIGHT_ID": "principal_verb",
        "RIGHT_ATTRS": {"POS": "AUX", "DEP": "ROOT", "LEMMA": "be"}
    },
    {
        "LEFT_ID": "principal_verb",
        "REL_OP": ">",
        "RIGHT_ID": "adj",
        "RIGHT_ATTRS": {"DEP": "acomp"}
    },
    {
        "LEFT_ID": "adj",
        "REL_OP": ">",
        "RIGHT_ID": "prep",
        "RIGHT_ATTRS": {"DEP": "prep", "text": "as"}
    },
    {
        "LEFT_ID": "prep",
        "REL_OP": ">",
        "RIGHT_ID": "simile_object",
        "RIGHT_ATTRS": {"DEP": "pobj"}
    },
    {
        "LEFT_ID": "principal_verb",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"DEP": {"IN":["nsubj", "csubj"]}}
    }
])
subj_indices.append(4)
simile_indices.append(3)

patterns.append([
    {
        "RIGHT_ID": "principal_verb",
        "RIGHT_ATTRS": {"POS": "AUX", "DEP": "ROOT", "LEMMA": "be"}
    },
    {
        "LEFT_ID": "principal_verb",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"DEP": {"IN":["nsubj", "csubj"]}}
    },
    {
        "LEFT_ID": "principal_verb",
        "REL_OP": ">",
        "RIGHT_ID": "simile_object",
        "RIGHT_ATTRS": {"DEP": "attr"}
    }
])
subj_indices.append(1)
simile_indices.append(2)

patterns.append([
    {
        "RIGHT_ID": "principal_verb",
        "RIGHT_ATTRS": {"POS": {"IN":["AUX", "VERB"]}, "DEP": "ROOT"}
    },
    {
        "LEFT_ID": "principal_verb",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"DEP": {"IN":["nsubj", "csubj"]}}
    },
    {
        "LEFT_ID": "principal_verb",
        "REL_OP": ">",
        "RIGHT_ID": "simile_object",
        "RIGHT_ATTRS": {"DEP": "attr"}
    }
])
subj_indices.append(1)
simile_indices.append(2)

patterns.append([
    {
        "RIGHT_ID": "principal_verb",
        "RIGHT_ATTRS": {"POS": {"IN":["VERB", "AUX"]}, "DEP": "ROOT"}
    },
    {
        "LEFT_ID": "principal_verb",
        "REL_OP": ">>",
        "RIGHT_ID": "like",
        "RIGHT_ATTRS": {"DEP": "prep", "text": "like"}
    },
    {
        "LEFT_ID": "like",
        "REL_OP": ">",
        "RIGHT_ID": "simile_object",
        "RIGHT_ATTRS": {"DEP": {"IN":["pobj", "pcomp"]}}
    },
    {
        "LEFT_ID": "principal_verb",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"DEP": {"IN":["nsubj", "csubj"]}}
    }
])
subj_indices.append(3)
simile_indices.append(2)

end_patterns = []
end_patterns.append([
    {
        "RIGHT_ID": "principal_verb",
        "RIGHT_ATTRS": {"POS": {"IN":["VERB", "AUX"]}, "DEP": "ROOT"}
    },
    {
        "LEFT_ID": "principal_verb",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass", "csubj"]}}
    }
])
end_patterns.append([
    {
        "RIGHT_ID": "principal_verb",
        "RIGHT_ATTRS": {"POS": "VERB", "DEP": "ROOT"}
    },
    {
        "LEFT_ID": "principal_verb",
        "REL_OP": ">>",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass", "csubj"]}}
    },
    {
        "LEFT_ID": "subject",
        "REL_OP": "<",
        "RIGHT_ID": "aux",
        "RIGHT_ATTRS": {"POS": "AUX"}
    }
])
end_subj_idx = 1

# It smells like a freshly baked cookies on Christmas morning; The movie had as much meaning as a Hallmark card The movie's meaning was frivolous. The movie was laden with meaning.
# The jazz solo sounded as smooth as silk The jazz solo sounded nice and smooth The jazz solo sounded rough and bad

doc = nlp("The girl is as easy to see through as a bowl of grape jelly")
# print([(token.dep_, token.pos_, token.head.text, token.lemma_) for token in doc])
# print(apply_template("The girl is as easy to see through as a bowl of grape jelly", "The girl is hard to figure out.", "The girl is easy to read.", patterns[1], end_patterns, 4, 3, 1))

match_count = 0
train = pd.read_csv("dev-combined.csv")     
print(train.shape)
results = []

for _, row in train.iterrows():
    start = row["startphrase"] # "The tree canopy has the shadow of a totem pole."
    ending1 = row["ending1"] # "The tree canopy does not provide shade."
    ending2 = row["ending2"] # "The tree canopy is shady."


    found_match = False
    for i in range(len(patterns)):
        pattern, subj_idx, simile_idx = patterns[i], subj_indices[i], simile_indices[i]
        object, modified1, modified2 = apply_template(start, ending1, ending2, pattern, end_patterns, subj_idx, simile_idx, end_subj_idx)
        if object != None:
            results.append({"startphrase": start,
                            "ending1": ending1,
                            "ending2": ending2,
                            "object": object,
                            "modified_1": modified1,
                            "modified_2": modified2,
                            "labels": row["labels"],
                            "qid": row["qid"],
                            "obj": row["obj"],
                            "vis": row["vis"],
                            "soc": row["soc"],
                            "cul": row["cul"],
                            "num_labels": row["num_labels"]})
            # print(start, ending1, ending2)
            # print("Object:", object, "; Phrase 1:", modified1, "; Phrase 2:", modified2)
            match_count += 1
            found_match = True
            break
    if not found_match: print(start, ending1, ending2)

print(match_count)
output_file_path = "rearranged_dev.csv"
output_df = pd.DataFrame(results)
output_df.to_csv(output_file_path, index=False)