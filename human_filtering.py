import pandas as pd


# Load the excel file as a DataFrame

df = pd.read_excel("/home/timkolber/news-please/cc_download_articles/nli_filtered/results.xlsx")

for index, row in df.iterrows():
    if row['Scores'] < 0.6:
        # Print the article title and first sentence
        print("\nFirst Sentence:\n", row['First Sentences'])
        print("\nTitle:\n", row['Titles'])
        
        # Prompt the user for the correct prediction
        correct_prediction = input("\nPlease enter the correct prediction (entailment:1/non-entailment:2): \n")
        
        if correct_prediction == '1':
            correct_prediction = 'entailment'
        
        elif correct_prediction == '2':
            correct_prediction = 'non-entailment'
        
        # Update the DataFrame with the correct prediction
        df.loc[index, 'Predictions'] = correct_prediction

# Save the DataFrame as an excel file

df.to_excel("/home/timkolber/news-please/cc_download_articles/nli_filtered/results.xlsx", index=False)