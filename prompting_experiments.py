# -*- coding: utf-8 -*-

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline, AutoTokenizer, AutoModelForSequenceClassification, StoppingCriteria
import transformers
import torch
import json
import textwrap
import os
import locale
import pandas as pd
import datasets
import locale
from transformers.pipelines.pt_utils import KeyDataset
from pythonrouge.pythonrouge import Pythonrouge
from evaluate import load
from datasets import Dataset

bertscore = load("bertscore")

import argparse

# Create a parser object
parser = argparse.ArgumentParser(description="Arguments to load for prompting experiments")

# Add arguments
parser.add_argument('--limit_samples', type=int, default=None, help='limit_samples')
parser.add_argument('--max_new_tokens', type=int, default=50, help='max_new_tokens')
parser.add_argument('--temp', type=float, default=0.7, help='temperature')
parser.add_argument('--top_p', type=float, default=0.95, help='top_p')
parser.add_argument('--rep_penalty', type=float, default=1.15, help='repetition_penalty')
parser.add_argument('--results', type=str, default='results', help='The file name for the results')
parser.add_argument('--prompt_mode', type=str, default='vanilla', help='The prompt mode to use')
parser.add_argument('--loop', action='store_true')
parser.add_argument('--model_name', type=str, default='TheBloke/stable-vicuna-13B-HF', help='The model name')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--ds', type=str, default=r"/pfs/data5/home/hd/hd_hd/hd_go226/bachelor/results.xlsx")

# Parse the command-line arguments
args = parser.parse_args()

locale.getpreferredencoding = lambda: "UTF-8"

# PARAMETERS
limit_samples = args.limit_samples
max_new_tokens = args.max_new_tokens
temperature = args.temp
top_p = args.top_p
repetition_penalty = args.rep_penalty
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_xlsx_path = args.ds

results_path = f"/pfs/data5/home/hd/hd_hd/hd_go226/bachelor/results/{args.results}.xlsx"


base_tokenizer = LlamaTokenizer.from_pretrained(args.model_name)

base_model = LlamaForCausalLM.from_pretrained(
    args.model_name,
    load_in_8bit=True,
    device_map='auto',
    offload_folder="./cache"
)

pipe = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=base_tokenizer,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    top_p=top_p,
    repetition_penalty=repetition_penalty
)



sample_dataset = datasets.load_from_disk("/pfs/data5/home/hd/hd_hd/hd_go226/bachelor/january_reuters_cleaned")

guidelines = "Headlines should be short and preferably snappy. They should come out of information in the body of the text and not present new information. Headlines are usually not in past tense; a headline about a past event is generally in present tense; one about a future event generally includes to (to meet, to decide, etc.) Within a publication section, headlines should be consistent; those that are mere labels shouldn’t be mixed with those that have verbs. Articles (a, an, the) are usually not used in headlines."

guidelines2 = """When writing a headline, follow these guidelines:
1. Is it in good taste? Anything offensive in any way? Can anything be taken a wrong way?
2. Does it attract the reader’s attention? How can it be improved without sacrificing 
accuracy?
3. Does it communicate clearly, quickly? Any confusion? Any odd words, double 
meanings?
4. Is it accurate, true? Proper words used? Is the thrust of subject-verb true?
5. A single “NO” above is a veto. One “No” vote represents thousands of readers. Start 
over: rethink the headline from the beginning."""

instruction = "Please make sure to be faithful to the sentence and not invent or exaggerate information in the headline."

def get_prompt(human_prompt, mode='vanilla', wrong_example=None, similar_examples=None):
    if wrong_example is not None:
        prompt_template = f'USER: Generate a headline for the following first sentence of a news article: \n"{human_prompt}"\n\nThe following headline is not good enough because it contains information that isn\'t in the first sentence of the news article: \n"{wrong_example}" \nASSISTANT:'
        return prompt_template
    if similar_examples is not None:
        prompt_template = f'USER: {similar_examples}\n\nGenerate a headline for the following first sentence of a news article:\n"{human_prompt}" \nASSISTANT:'
        return prompt_template
    elif mode == "random-fewshot":
        shuffled_dataset = sample_dataset.shuffle()
        shuffled_dataset = sample_dataset.select(range(3))
        prompt_template = f'USER: Article: {shuffled_dataset[0]["First Sentences"]}\nHeadline: {shuffled_dataset[0]["Titles"]}\n\nArticle: {shuffled_dataset[1]["First Sentences"]}\nHeadline: {shuffled_dataset[1]["Titles"]}\n\nArticle: {shuffled_dataset[2]["First Sentences"]}\nHeadline: {shuffled_dataset[2]["Titles"]}\n\nGenerate a headline for the following first sentence of a news article:\n"{human_prompt}" \nASSISTANT:'
        return prompt_template
    elif mode == "guidelines":
        prompt_template=f'USER: {guidelines} \n\nGenerate a headline for the following first sentence of a news article: \n"{human_prompt}" \nASSISTANT:'
        return prompt_template
    elif mode == "instruction":
        prompt_template=f'USER: {instruction} \n\nGenerate a headline for the following first sentence of a news article: \n"{human_prompt}" \nASSISTANT:'
        return prompt_template
    prompt_template=f'USER: Generate a headline for the following first sentence of a news article:"{human_prompt}" \nASSISTANT:'
    return prompt_template

    
def apply_get_prompt(first_sentence):
    prompt = get_prompt(first_sentence, args.prompt_mode)
    return prompt

def remove_human_text(text):
    return text.split('USER:', 1)[0]

def remove_final_punctuation(string):
    if len(string) == 0:
        return string
    if string[-1] == ".":
        string = string[:-1]
    return string

def remove_quotes(string):
    if len(string) == 0:
        return string
    if string[0] == '"':
        string = string[1:]
    if string[-1] == '"':
        string = string[:-1]
    return string

def parse_text(data):
    for item in data:
        text = item['generated_text']
        assistant_text_index = text.find('ASSISTANT:')
        if assistant_text_index != -1:
            assistant_text = text[assistant_text_index+len('ASSISTANT:'):].strip()
            assistant_text = remove_human_text(assistant_text)
            assistant_text = remove_quotes(assistant_text)
            assistant_text = remove_final_punctuation(assistant_text)
            return(assistant_text)



deberta_model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
label_names = ["entailment", "non-entailment"]

device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(deberta_model_name)

model = AutoModelForSequenceClassification.from_pretrained(deberta_model_name).to(device)

df = pd.read_excel(dataset_xlsx_path)
if limit_samples:
    examples = df.head(limit_samples)

if args.resume:
    old_results_df = pd.read_excel(results_path)
    last_gold_title = str(old_results_df['Gold Titles'].iloc[-1])
    index = df[df['Titles'] == last_gold_title].index[0]
    examples = df.iloc[index + 1:]
else:
    examples = df

    
titles = examples["Titles"].tolist()
first_sentences = examples["First Sentences"].tolist()
dataset = datasets.Dataset.from_pandas(examples)
examples["Prompts"] = examples["First Sentences"].apply(apply_get_prompt)
dataset = datasets.Dataset.from_pandas(examples)
pipe.tokenizer.pad_token_id = 2

rouge_1 = []
rouge_2 = []
rouge_su4 = []
bert_scores = []
generated_titles = []
gold_titles=[]
first_sentences=[]
bert_scores=[]


for out, datapoint in zip(pipe(KeyDataset(dataset, "Prompts")), dataset):
    generated_title = parse_text(out)
    if args.loop:
    	for _ in range(5):
            input = tokenizer([datapoint['First Sentences']], [generated_title], truncation=True, padding=True, return_tensors="pt")
            output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
            instance = output['logits'][0]
            prediction = torch.softmax(instance, -1).tolist()
            # change labels to entailment or non_entailment
            prediction = [prediction[0], prediction[1] + prediction[2]]
            prediction_name = label_names[prediction.index(max(prediction))]
            if prediction_name == 'entailment':
                break
            else:
                prompt = get_prompt(human_prompt=datapoint['First Sentences'], wrong_example=generated_title)
                out = pipe(prompt)
                generated_title = parse_text(out)
                
    elif args.prompt_mode == "fewshot":
        prompt = get_prompt(human_prompt=datapoint['First Sentences'], similar_examples=datapoint["top_3_samples"])
        print(prompt)
        out = pipe(prompt)
        generated_title = parse_text(out)
    generated_titles.append(generated_title)
    gold_title = datapoint["Titles"]
    gold_titles.append(gold_title)
    first_sentence = datapoint["First Sentences"]
    first_sentences.append(first_sentence)
    summary = [[generated_title]]
    reference = [[[gold_title]]]
    rouge = Pythonrouge(summary_file_exist=False,
                    summary=summary, reference=reference,
                    n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                    recall_only=False, stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=50,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
    score = rouge.calc_score()
    rouge_1.append(score["ROUGE-1-F"])
    rouge_2.append(score["ROUGE-2-F"])
    rouge_su4.append(score["ROUGE-SU4-F"])
    pred = [generated_title]
    ref = [gold_title]
    bert_score = bertscore.compute(predictions=pred, references=ref, lang="en", device="cuda:0", batch_size=1)
    bert_score = bert_score["f1"][0]
    bert_scores.append(bert_score)
    data = {"First sentences": first_sentences, "Gold Titles": gold_titles, "Generated Titles": generated_titles, "Rouge-1": rouge_1, "Rouge-2": rouge_2, "Rouge-SU4": rouge_su4, "Bert-Score": bert_scores}
    results_df = pd.DataFrame(data=data)
    if args.resume:
        results_df = pd.concat([old_results_df, results_df], ignore_index=True)
    results_df.to_excel(results_path, index=False)

if args.resume:
    results_df = pd.read_excel(results_path)
results_df = results_df.fillna('')
device = torch.device("cpu")
batch_size = 4
titles = results_df["Generated Titles"].tolist()
first_sentences = results_df["First sentences"].tolist()
# Split hypotheses and premises into batches
titles_batches = [titles[i:i+batch_size] for i in range(0, len(titles), batch_size)]
first_sentences_batches = [first_sentences[i:i+batch_size] for i in range(0, len(first_sentences), batch_size)]


# Perform batch inference
prediction_names = []
for titles_batch, first_sentence_batch in zip(titles_batches, first_sentences_batches):
    input = tokenizer(first_sentence_batch, titles_batch, truncation=True, padding=True, return_tensors="pt")
    # label_names = ["entailment", "neutral", "contradiction"]
    output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    for instance in output["logits"]:
        prediction = torch.softmax(instance, -1).tolist()
        # change labels to entailment or non_entailment
        prediction = [prediction[0], prediction[1] + prediction[2]]
        prediction_name = label_names[prediction.index(max(prediction))]
        prediction_names.append(prediction_name)

results_df["Deberta Prediction"] = prediction_names

results_df.to_excel(results_path, index=False)
