# -*- coding: utf-8 -*-
"""fig-llama-test.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1z3mZmdrfVPjw8YJIVTlEyEGvz-i5K1OJ
"""

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfFolder

!huggingface-cli login

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = 'meta-llama/Llama-3.2-1B'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def get_prob(sentence):

    tokens = tokenizer.encode(sentence)
    tokens_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    total_log_prob = 0.0

    start_ind = 1

    for i in range(start_ind, len(tokens)):

        input_tokens = tokens[:i]
        target_token = tokens[i]
        input_tensor = torch.tensor(input_tokens).unsqueeze(0).to(device)

        outputs = model(input_tensor)
        logits = outputs.logits

        logits_last = logits[0, -1, :]
        probs = F.softmax(logits_last, dim=-1)
        target_prob = probs[target_token].item()

        total_log_prob += torch.log(torch.tensor(target_prob))

    sentence_probability = torch.exp(total_log_prob).item() / len(tokens)

    return sentence_probability

input_path = 'rearranged_dev_filtered.csv'
df = pd.read_csv(input_path, delimiter=',')

def process_row(row):

    sent_1 = row['modified_1'].rstrip('.') + '.'
    sent_2 = row['modified_2'].rstrip('.') + '.'

    prob_1 = get_prob(sent_1)
    prob_2 = get_prob(sent_2)

    result = int((prob_1 < prob_2) == row['labels'])

    return result

df['result'] = df.apply(process_row, axis=1)

output_path = 'fig-llama-results.csv'
df.to_csv(output_path, sep=',', index=False)

print(df['result'].mean())