import pandas as pd
import os
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from datasets import load_from_disk, Dataset

device = torch.device("cpu")
batch_size = 8
# Set the paths to your source and destination directories
source_directory = r"/home/timkolber/news-please/cc_download_articles/www.reuters.com"
destination_directory = r"/home/timkolber/news-please/cc_download_articles/nli_filtered"


# Get the list of all JSON files in the source directory
json_files = [file for file in os.listdir(source_directory) if file.endswith(".json")]

# Create a list to store the extracted data
data = []

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

first_sentences = []
titles = []
# Extract "title" and "description" from each file
for index, file in enumerate(json_files):
    file_path = os.path.join(source_directory, file)
    with open(file_path, "r", encoding="utf-8") as f1:
        article = json.load(f1)
        if article["title"] is None or article["description"] is None:
            print(f"File {index} is empty")
            continue
        titles.append(article["title"])
        first_sentences.append(article["description"])        

# Move the selected files to the destination directory
for file in json_files:
    source_file = os.path.join(source_directory, file)
    destination_file = os.path.join(destination_directory, file)
    shutil.copy(source_file, destination_file)

# Split hypotheses and premises into batches
first_sentences_batches = [titles[i:i+batch_size] for i in range(0, len(titles), batch_size)]
titles_batches = [first_sentences[i:i+batch_size] for i in range(0, len(first_sentences), batch_size)]
        
deberta_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
label_names = ["entailment", "non-entailment"]

model_predictions = {}

tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)
model = AutoModelForSequenceClassification.from_pretrained(deberta_model_name).to(device)
# Perform batch inference
prediction_names = []
prediction_scores = []
for titles_batch, first_sentence_batch in zip(titles_batches, first_sentences_batches):
    input = tokenizer(titles_batch, first_sentence_batch, truncation=True, padding=True, return_tensors="pt")
    # label_names = ["entailment", "neutral", "contradiction"]
    output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    for instance in output["logits"]:
        prediction = torch.softmax(instance, -1).tolist()
        # change labels to entailment or non_entailment
        prediction = [prediction[0], prediction[1] + prediction[2]]
        prediction_name = label_names[prediction.index(max(prediction))]
        prediction_score = max(prediction)
        prediction_names.append(prediction_name)
        prediction_scores.append(prediction_score)
        

data = {
    "First Sentences": first_sentences,
    "Titles": titles,
    "Predictions": prediction_names,
    "Scores": prediction_scores
}



df = pd.DataFrame(data)

filtered_df = df[(df["Predictions"] == "entailment")]

# Calculate the length of titles and first sentences
filtered_df['Title Length'] = filtered_df['Titles'].apply(len)
filtered_df['Sentence Length'] = filtered_df['First Sentences'].apply(len)

# Set a threshold for significant difference in lengths
threshold = 20

# Filter out rows where titles are not significantly shorter than first sentences
filtered_df = filtered_df[filtered_df['Title Length'] + threshold < filtered_df['Sentence Length']]

hf_dataset = Dataset.from_pandas(filtered_df)

hf_dataset = hf_dataset.filter(lambda example: example['Titles'] != None)
hf_dataset = hf_dataset.filter(lambda example: example['First Sentences'] != None)
hf_dataset = hf_dataset.filter(lambda example: "-" not in example['Titles'])
hf_dataset = hf_dataset.filter(lambda example: ":" not in example['Titles'])
hf_dataset = hf_dataset.filter(lambda example: "?" not in example['Titles'])

hf_dataset.save_to_disk("/home/timkolber/news-please/august_reuters")

filtered_df = hf_dataset.to_pandas()

filtered_df.to_excel(r"/home/timkolber/news-please/cc_download_articles/nli_filtered/results.xlsx")
