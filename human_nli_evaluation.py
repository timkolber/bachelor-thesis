import json
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

annotations_file_path = r"/home/timkolber/news-please/cc_download_articles/www.reuters.com/random_selection/to_annotate/annotations.json"
results_file_path = r"/home/timkolber/news-please/cc_download_articles/www.reuters.com/random_selection/results.xlsx"
human_labels = {}
model_names = ["Deberta", "Roberta", "ChatGPT"]

with open(annotations_file_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
        
for article in articles:
    title = article["text"].split("TITLE:\n'")[-1]
    title = title[:-1]
    label = article["label"][0].lower()
    human_labels[title] = label

df = pd.read_excel(results_file_path)
df["Gold label"] = df["Titles"].map(human_labels)
df.to_excel(results_file_path, index=False)

for model in model_names:
    print(model)
    ConfusionMatrixDisplay.from_predictions(df["Gold label"], df[f"Predictions {model}"])
    plt.show()

    print("Metrics for Entailment Class:")
    print('Precision: %.3f' % precision_score(df["Gold label"], df[f"Predictions {model}"], pos_label="entailment"))
    print('Recall: %.3f' % recall_score(df["Gold label"], df[f"Predictions {model}"], pos_label="entailment"))
    print('F1 Score: %.3f' % f1_score(df["Gold label"], df[f"Predictions {model}"], pos_label="entailment"))

    print()

    print("Metrics for Non-Entailment Class:")
    print('Precision: %.3f' % precision_score(df["Gold label"], df[f"Predictions {model}"], pos_label="non-entailment"))
    print('Recall: %.3f' % recall_score(df["Gold label"], df[f"Predictions {model}"], pos_label="non-entailment"))
    print('F1 Score: %.3f' % f1_score(df["Gold label"], df[f"Predictions {model}"], pos_label="non-entailment"))