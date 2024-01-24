import pandas as pd
import os
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from datasets import load_from_disk, Dataset

device = torch.device("cuda:0")
batch_size = 8
# Set the paths to your source and destination directories
source_directory = r"/home/timkolber/news-please/august_reuters/"
destination_directory = r"/home/timkolber/news-please/august_reuters_cleaned"

hf_dataset = load_from_disk(source_directory)

print(len(hf_dataset))

deberta_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
label_names = ["entailment", "non-entailment"]
model_predictions = {}

tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)
model = AutoModelForSequenceClassification.from_pretrained(deberta_model_name).to(device)

prediction_names = []
prediction_scores = []

print("Number of samples in the dataset:")
print(len(hf_dataset))

hf_dataset = hf_dataset.filter(lambda example: example['title'] != None)
hf_dataset = hf_dataset.filter(lambda example: example['first_sentence'] != None)
hf_dataset = hf_dataset.filter(lambda example: "-" not in example['title'])
hf_dataset = hf_dataset.filter(lambda example: ":" not in example['title'])
hf_dataset = hf_dataset.filter(lambda example: "?" not in example['title'])

for article in hf_dataset:
    first_sentence = article["first_sentence"]
    title = article["title"]
    input = tokenizer(first_sentence, title, truncation=True, padding=True, return_tensors="pt")
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

data = {"Titles": hf_dataset["title"],
        "First Sentences": hf_dataset["first_sentence"],
        "Predictions": prediction_names,
        "Scores": prediction_scores}
df = pd.DataFrame(data)

df.to_excel(r"/home/timkolber/news-please/august_articles/nli_filtered/results_unfiltered.xlsx")


filtered_df = df[(df["Predictions"] == "entailment")]



# Calculate the length of titles and first sentences
filtered_df['Title Length'] = filtered_df['Titles'].apply(len)
filtered_df['Sentence Length'] = filtered_df['First Sentences'].apply(len)

filtered_df.to_excel(r"/home/timkolber/news-please/august_articles/nli_filtered/results.xlsx")
