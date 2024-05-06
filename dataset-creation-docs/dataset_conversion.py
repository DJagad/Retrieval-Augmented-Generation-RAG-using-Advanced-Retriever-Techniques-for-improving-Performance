import json
import random

def convert_jsonl_to_qa_format(jsonl_file):
    qa_pairs = []

    with open(jsonl_file, 'r') as f:
        for line in f:
            entry = json.loads(line)

            if "instruction" in entry and "response" in entry:
                question = entry["instruction"]
                answer = entry["response"]
                qa_pair = {
                    "question": question,
                    "answer": answer
                }
                qa_pairs.append(qa_pair)
    
    return qa_pairs

def write_qa_pairs_to_file(qa_pairs, output_file):
    with open(output_file, 'w') as f:
        for i, pair in enumerate(qa_pairs, start=1):
            f.write(f"Q{i}: {pair['question']}\n")
            f.write(f"A{i}: {pair['answer']}\n\n")

# Replace 'input.jsonl' with the path to your JSON Lines file
jsonl_file = 'databricks-dolly-15K.jsonl'
output_file = 'databricks-dolly-15K.txt'

qa_pairs = convert_jsonl_to_qa_format(jsonl_file)

# Shuffle the QA pairs and select 300 randomly
random.shuffle(qa_pairs)
qa_pairs = qa_pairs[:150]

# Write the QA pairs to the output file
write_qa_pairs_to_file(qa_pairs, output_file)

print("Conversion completed. Output written to 'databricks-dolly-15K.txt'")