import json
import random

# Load the original data
original_data = []
with open('databricks-dolly-15k.jsonl', 'r') as file:
    for line in file:
        original_data.append(json.loads(line.strip()))

# Shuffle the data randomly
random.shuffle(original_data)

# Take the first 300 lines
sampled_data = original_data[:900]

# Write the sampled data to a new JSONL file
with open('sampled_data.jsonl', 'w') as file:
    for item in sampled_data:
        json.dump(item, file)
        file.write('\n')

print(f"Sampled {len(sampled_data)} lines from the original data and saved to 'sampled_data.jsonl'")