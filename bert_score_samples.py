from evaluate import load
from datasets import load_from_disk, Dataset
import argparse
import numpy as np
import time

argparser = argparse.ArgumentParser()
argparser.add_argument('--sample_ds_path', type=str)
argparser.add_argument('--ds_path')
argparser.add_argument('--output_path', type=str, default='.')
argparser.add_argument('--n', type=int, default='3')
args = argparser.parse_args()

samples = load_from_disk(args.sample_ds_path)
dataset = load_from_disk(args.ds_path)

bertscore = load('bertscore')
top_n_feature = []

for datapoint in dataset:
    bertscores_samples = []
    first_sentences_samples = []
    titles_samples = []
    for sample in samples:
        titles_samples.append(sample['title'])
        first_sentences_samples.append(sample['first_sentence'])
        bertscore_sample = bertscore.compute(predictions=[sample['first_sentence']], references=[datapoint['first_sentence']], lang='en')
        bertscore_sample_f1 = bertscore_sample['f1']
        bertscores_samples.append(bertscore_sample_f1)
    top_n_first_sentences = sorted(zip(bertscores_samples, first_sentences_samples), key=lambda x: x[0],  reverse=True)[:args.n]
    top_n_titles = sorted(zip(bertscores_samples, titles_samples), key=lambda x: x[0],  reverse=True)[:args.n]
    top_n_first_sentences = [x[1] for x in top_n_first_sentences]
    top_n_titles = [x[1] for x in top_n_titles]
    top_n_string = ""
    for first_sentence, title in zip(top_n_first_sentences, top_n_titles):
        top_n_string += f"Article: {first_sentence}\nHeadline: {title}\n\n"
    top_n_string = top_n_string[:-2]
    top_n_feature.append(top_n_string)

dataset = dataset.add_column(f'top_{args.n}_samples', top_n_feature)
dataset.save_to_disk(args.output_path)
