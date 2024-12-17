import spacy
import pandas as pd
from spacy.matcher import DependencyMatcher

nlp = spacy.load('en_core_web_sm')

singular_noun_tags = ["NN", "NNP"]
plural_noun_tags = ["NNS", "NNPS"]
singular3p_pronouns = ["HE", "SHE", "IT"]
singular3p_verb_tags = ["MD", "AUX", "VBD", "VBZ"]
non3p_verb_tags = ["MD", "AUX", "VBD", "VBP"]

def apply_template(start_phrase, ending1, ending2, start_pattern, end_patterns, subj_idx, simile_idx, end_subj_idx):
    start_doc = nlp(start_phrase)
    start_matcher = DependencyMatcher(nlp.vocab)
    start_matcher.add("start", [start_pattern])
    start_matches = start_matcher(start_doc)
    if len(start_matches) != 1:
        return None, None, None, None

    indices = start_matches[0][1]
    subject = start_doc[indices[subj_idx]].lemma
    simile_np = [token for token in start_doc[indices[simile_idx]].subtree]

    end_doc1, end_doc2 = nlp(ending1), nlp(ending2)
    end_matcher = DependencyMatcher(nlp.vocab)
    end_matcher.add("end", end_patterns)
    end_matches1, end_matches2 = end_matcher(end_doc1), end_matcher(end_doc2)
    if len(end_matches1) != 1 or len(end_matches2) != 1:
        return None, None, None, None
    
    indices1, indices2 = end_matches1[0][1], end_matches2[0][1]
    end_subj1, end_subj2 = end_doc1[indices1[end_subj_idx]], end_doc2[indices2[end_subj_idx]]
    
    if (end_subj1.lemma != subject and end_subj1.lemma_ != "it" and end_subj1.pos_ != "PRON") or (end_subj2.lemma != subject and end_subj2.lemma_ != "it" and end_subj2.pos_ != "PRON"):
        # print(start_doc[indices[subj_idx]].lemma_, end_subj1.lemma_) # ^^^ dealing with pronoun
        return None, None, None, None
    
    # check agreement
    agrees = [0, 0] # -1 : incorrect, 0 : undetermined, 1 : correct
    subj_tag = start_doc[indices[simile_idx]].tag_
    verbs = [end_doc1[indices1[0]], end_doc2[indices2[0]]]
    for i in range(2):
        for child in verbs[i].children:
            if child.dep_ in ["aux", "auxpass"]:
                verbs[i] = child
                break

    if subj_tag in singular_noun_tags or (subj_tag == "PRP" and start_doc[indices[subj_idx]] in singular3p_pronouns):
        for i in range(2):
            verb = verbs[i]
            end = end_doc1 if i == 0 else end_doc2
            if verb.tag_ in singular3p_verb_tags: agrees[i] = 1
            elif verb.tag_ == "VBP": agrees[i] = -1
            else: print(verb, verb.tag_, start_doc[indices[simile_idx]], subj_tag, end, [(child, child.dep_) for child in verb.children])
    else:
        for i in range(2):
            verb = verbs[i]
            end = end_doc1 if i == 0 else end_doc2
            if verb.tag_ in non3p_verb_tags: agrees[i] = 1
            elif verb.tag_ == "VBZ": agrees[i] = -1
            else: print(verb, verb.tag_, start_doc[indices[simile_idx]], subj_tag, end, [(child, child.dep_) for child in verb.children])
    
    np1, np2 = [token for token in end_subj1.subtree], [token for token in end_subj2.subtree]
    
    mod1 = end_doc1[:np1[0].i].text_with_ws
    for token in simile_np: mod1 += token.text_with_ws
    mod1 = mod1.rstrip() + end_doc1[np1[-1].i].whitespace_ + end_doc1[np1[-1].i+1:].text_with_ws

    mod2 = end_doc2[:np2[0].i].text_with_ws
    for token in simile_np: mod2 += token.text_with_ws
    mod2 = mod2.rstrip() + end_doc2[np2[-1].i].whitespace_ + end_doc2[np2[-1].i+1:].text_with_ws

    return "".join([token.text_with_ws for token in simile_np]), mod1.capitalize(), mod2.capitalize(), agrees

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

# doc = nlp("The girl is as easy to see through as a bowl of grape jelly")
# print([(token.dep_, token.pos_, token.tag_, token.head.text, token.lemma_) for token in doc])

match_count = 0
data = pd.read_csv("dev.csv")     
print(data.shape)
results = []

agreement_count = [0, 0, 0]

for _, row in data.iterrows():
    start = row["startphrase"]
    ending1 = row["ending1"]
    ending2 = row["ending2"]


    found_match = False
    for i in range(len(patterns)):
        pattern, subj_idx, simile_idx = patterns[i], subj_indices[i], simile_indices[i]
        object, modified1, modified2, agrees = apply_template(start, ending1, ending2, pattern, end_patterns, subj_idx, simile_idx, end_subj_idx)
        if object != None:
            agreement_count[agrees[0]+1] += 1
            agreement_count[agrees[1]+1] += 1

            if agrees[0] == 1 and agrees[1] == 1:
                root_verb1, root_verb2 = nlp(modified1)[0].sent.root.text, nlp(modified2)[0].sent.root.text 
                results.append({"startphrase": start, "ending1": ending1, "ending2": ending2, "object": object, "modified_1": modified1, "modified_2": modified2, "verb_1": root_verb1, "verb_2": root_verb2, "labels": row["labels"], "qid": row["qid"]})
                match_count += 1

# print(agreement_count)
print("Found", match_count, "matches.")
output_file_path = "rearranged_dev_filtered.csv"
output_df = pd.DataFrame(results)
output_df.to_csv(output_file_path, index=False)
