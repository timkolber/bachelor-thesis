import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

device = torch.device("cpu")
batch_size = 4
file_path = r"C:\Users\timko\Desktop\news-please\cc_download_articles\www.reuters.com\random_selection\extracted_data.json"

with open(file_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
        
first_sentences = []
titles = []
for article in articles:
    first_sentences.append(article["description"])
    titles.append(article["title"])
    
# Split hypotheses and premises into batches
first_sentences_batches = [titles[i:i+batch_size] for i in range(0, len(titles), batch_size)]
titles_batches = [first_sentences[i:i+batch_size] for i in range(0, len(first_sentences), batch_size)]
        
deberta_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
roberta_model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
label_names = ["entailment", "non-entailment"]
model_names = [deberta_model_name, roberta_model_name]

model_predictions = {}
for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
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
    model_predictions[model_name] = (prediction_names, prediction_scores)

data = {
    "First Sentences": first_sentences,
    "Titles": titles,
    "Predictions Deberta": model_predictions[deberta_model_name][0],
    "Prediction Scores Deberta": model_predictions[deberta_model_name][1],
    "Predictions Roberta": model_predictions[roberta_model_name][0],
    "Prediction Scores Roberta": model_predictions[roberta_model_name][1],
}

df = pd.DataFrame(data)
df.to_excel(r"C:\Users\timko\Desktop\news-please\cc_download_articles\www.reuters.com\random_selection\results.xlsx")
