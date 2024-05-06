import json
import random

# Load the dataset from the JSONL file
dataset = []
with open("sampled_data.jsonl", "r") as f:
    for line in f:
        dataset.append(json.loads(line))

# Randomly select 50 questions from the dataset
random_questions = random.sample(dataset, 10)

# Create a new list to store the test dataset
test_dataset = []

# Iterate over the randomly selected questions and extract the questions
for data_point in random_questions:
    question = {
        "instruction": data_point["instruction"],
        "response": data_point["response"]
    }
    test_dataset.append(question)

# Save the test dataset as a JSONL file
with open("test_dataset.jsonl", "w") as f:
    for question in test_dataset:
        json_string = json.dumps(question)
        f.write(json_string + "\n")