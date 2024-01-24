import pandas as pd
import glob

from datasets import Dataset
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--directory_path', type=str, default='.')
argparser.add_argument('--output_path', type=str, default='.')
argparser.add_argument('--n', type=int, default='-1')
args = argparser.parse_args()

# Get a list of JSON file paths in the directory
json_files = glob.glob(args.directory_path + '/*.json')

if (args.n == -1):
    n = len(json_files)

# Initialize an empty list to store the dataframes
dfs = []

# Iterate over each JSON file and load it into a dataframe
for file in json_files:
    df = pd.read_json(file)
    dfs.append(df)

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.drop(['authors', 'date_download', 'date_modify', 'filename', 'image_url', 'localpath', 'source_domain', 'title_page', 'title_rss', 'url', 'maintext'], axis=1, inplace=True)
combined_df.rename(columns={'description': 'first_sentence'}, inplace=True)
combined_df = combined_df.drop_duplicates()
dataset = Dataset.from_pandas(combined_df)
shuffled_dataset = dataset.shuffle(seed=42)
if (args.n != -1):
    shuffled_dataset = shuffled_dataset.select(range(args.n))
shuffled_dataset.save_to_disk(args.output_path)

print("Number of samples in the dataset:")
print(len(shuffled_dataset))


