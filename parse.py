import spacy
import pandas as pd
from spacy.matcher import DependencyMatcher

nlp = spacy.load('en_core_web_sm')
start_matcher = DependencyMatcher(nlp.vocab)
pattern = [
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
        "RIGHT_ATTRS": {"DEP": "pobj", "POS": "NOUN"}
    },
    {
        "LEFT_ID": "principal_verb",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"DEP": "nsubj", "POS": "NOUN"}
    }
    ]
start_matcher.add("pattern1", [pattern])

start = "The tree canopy has the shadow of a totem pole."
endings = "The tree canopy does not provide shade. The tree canopy is shady."

doc = nlp(start)
matches = start_matcher(doc)
phrase1 = [0, 0]
subj1 = ""
phrase2 = [0, 0]
subj2 = ""
if len(matches) > 0:
    for match in matches:
        subject = doc[match[1][4]]
        subj1 = subject.text
        simile_object = doc[match[1][3]]
        subj2 = simile_object.text
        print(subj1, subj2)

        # nphrase1 = [item for item in subject.subtree]
        nphrase2 = [item for item in simile_object.subtree]
        # phrase1 = [nphrase1[0].i, nphrase1[-1].i]
        phrase2 = [nphrase2[0].i, nphrase2[-1].i] # indices bounding simile object noun phrase
        print(doc)

doc2 = nlp(endings)
pattern = [
    {
        "RIGHT_ID": "principal_verb",
        "RIGHT_ATTRS": {"POS": "VERB", "DEP": "ROOT"}
    },
    {
        "LEFT_ID": "principal_verb",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"text": subj1, "DEP": {"IN": ["nsubj", "nsubjpass"]}}
    }
]

end_matcher = DependencyMatcher(nlp.vocab)
end_matcher.add("pattern2", [pattern])
matches = end_matcher(doc2)
sentences = []
if len(matches) > 0:
    for match in matches:
        subject = doc2[match[1][1]]
        noun_phrase = [item for item in subject.subtree]
        whitespace = ""
        new_sentence = ""
        first_word = True
        for token in doc2[:noun_phrase[0].i]:
            new_sentence += whitespace + token.text
            whitespace = token.whitespace_
        for token in doc[phrase2[0]:phrase2[1]+1]:
            new_sentence += whitespace + token.text
            whitespace = token.whitespace_
        whitespace = doc2[noun_phrase[-1].i].whitespace_
        for token in doc2[noun_phrase[-1].i+1:]:
            new_sentence += whitespace + token.text
            whitespace = token.whitespace_
        sentences.append(new_sentence[0].upper() + new_sentence[1:])

print(sentences)

# train = pd.read_csv("train.csv")     
# print(train.shape)

doc = nlp("The dog was as grumpy as a kindergarten teacher.")
for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text, token.lemma_)
    print([item.text for item in token.subtree])
# doc = nlp("Autonomous cars shift insurance liability toward manufacturers")
# for index, row in train[295:300].iterrows():
#     startphrase, end1, end2, label, valid, qid = row
#     print("Start:", startphrase, "; end1:", end1, "; end2:", end2, "; label:", label)
#     doc = nlp(startphrase)
#     for token in doc:
#         print(token.text, token.pos_, token.dep_)#, token.head.text, token.head.pos_,
            #[child for child in token.children])

# doc1 = nlp("The team's defense has as many holes in it as a Qanon conspiracy theory.")
# for token in doc1:
#     print(token.text, token.pos_, token.dep_)

# doc2 = nlp("The tree canopy has the shadow of a totem pole.")
# for token in doc2:
#     print(token.text, token.pos_, token.dep_, token.lemma_)
