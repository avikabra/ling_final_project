import pandas as pd
import requests
import json

file_path = "train.csv"
data = pd.read_csv(file_path)

def query_ollama(model: str, prompt: str):
    url = f"http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Error querying Ollama: {response.text}")

prompt_template = """
You are an expert at restructuring sentences. Given the following metaphorical sentence, focus on the object of comparison.
Rewrite the two endings provided to emphasize the object instead of the original subject.

Input sentence: "{sentence}"
Ending 1: "{ending1}"
Ending 2: "{ending2}"

Task:
1. Extract the object of comparison from "{sentence}".
2. Rewrite Ending 1 and Ending 2 to focus on the object of comparison.

Output format:
- Object: <object>
- Ending 1: <rewritten ending 1>
- Ending 2: <rewritten ending 2>
"""

results = []
for _, row in data.iterrows():
    prompt = prompt_template.format(
        sentence=row["startphrase"],
        ending1=row["ending1"],
        ending2=row["ending2"]
    )
    try:
        response = query_ollama(model="llama2", prompt=prompt)
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        
        object_ = ""
        ending1 = ""
        ending2 = ""
        
        for line in lines:
            if line.startswith("Object:"):
                object_ = line.replace("Object:", "").strip()
            elif line.startswith("Ending 1:"):
                ending1 = line.replace("Ending 1:", "").strip()
            elif line.startswith("Ending 2:"):
                ending2 = line.replace("Ending 2:", "").strip()
        
        results.append({"object": object_, "ending_1": ending1, "ending_2": ending2})
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        results.append({"object": "", "ending_1": "", "ending_2": ""})

output_file_path = "rearranged_sentences_ollama.csv"
output_df = pd.DataFrame(results)
output_df.to_csv(output_file_path, index=False)

print(f"Rewritten sentences saved to {output_file_path}")
