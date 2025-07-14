from datasets import load_dataset
from utils import data_loader
from datasets import Dataset
import os
import pandas as pd

def preprocess_data():
    # Load the data
    # raw_data_columns = [Vignette,Question,Answer,Dosage,Explanation]
    df = data_loader.load_raw_data()

    result = []

    df.columns = [col.lower() for col in df.columns]

    df.rename(columns={
                       'answer': 'answer_bool',
                       'dosage': 'answer_dosage'},
            inplace=True)
    df['answer_bool'] = df['answer_bool'].str.lower().str.replace('.', '')
    
    
    df['pain_type'] = df['source_file'].apply(lambda x: x.split('.')[0].replace('data_', '').replace('_', ' '))

    # Split the 'question' column by '?' and extract parts
    def parse_question(q):
        parts = q.split('?')
        drug = parts[0].strip() if len(parts) > 0 else ''
        option = parts[1].strip() if len(parts) > 1 else ''
        dosage_option = parts[2].strip() if len(parts) > 2 else ''
        return pd.Series([drug, option, dosage_option])

    df[['question_drug', 'question_option', 'question_dosage']] = df['question'].apply(parse_question)

    df['question_drug'] = df['question_drug'].str.replace(r'Would you offer (a )?', '', regex=True)
    df['question_drug'] = df['question_drug'].str.replace(r'(to )?Patient [BD]( )?', '', regex=True)
    df['question_drug'] = df['question_drug'].str.replace(r'( )?for pain control( )?', '', regex=True)
    df['question_drug'] = df['question_drug'].str.replace(r'( )?prescription( )?', '', regex=True)
    df['question_drug'] = df['question_drug'].str.strip()

    df['question_dosage'] = df['question_dosage'].str.replace('If yes, how much supply would you give – ', '', regex=False)
    df['question_dosage'] = df['question_dosage'].str.replace('If yes, what dose – ','', regex=False)
    df['question_dosage'] = df['question_dosage'].str.replace(' supply', '', regex=False)
    df['question_dosage'] = df['question_dosage'].str.lower()
    df['question_dosage'] = df['question_dosage'].str.replace(r'(,)? or ',',', regex=True)
    df['question_dosage'] = df['question_dosage'].str.replace('weeks', 'week', regex=False)

    df[['question_drug','pain_type','answer_bool']].value_counts()
    # df.groupby(['pain_type','question_drug','answer_bool']).size()

    # df.groupby(['question_drug','pain_type','answer_bool']).size()
    
    # return df[['instruction','question', 'answer']].to_dict(orient='records')
    col = ['vignette', 'answer_bool', 'answer_dosage',
       'explanation', 'pain_type', 'question_drug',
       ]
    return df[col]

def _format_answer(r):

    short_answer = r['Answer']

    if short_answer == 'Yes':
        short_answer += f", with dosage {r['Dosage']}"

    return f"{short_answer}. {r['Explanation']}"

df = preprocess_data()
# df.to_csv('local/data/processed/preprocessed_data.csv', index=False)

# # Convert pandas DataFrame to HuggingFace Dataset
# df.rename(columns={'question': 'input',
#                    'answer': 'output'}, inplace=True)
dataset = Dataset.from_pandas(df)

# First split: train + temp (test+eval)
train_test = dataset.train_test_split(test_size=0.1, seed=42)  # 70% train, 30% temp

# Second split: test + eval from temp
test_eval = train_test['test'].train_test_split(test_size=0.5, seed=42)  # 15% test, 15% eval

# Combine all splits into a DatasetDict
from datasets import DatasetDict

splits = DatasetDict({
    'train': train_test['train'],
    'test': test_eval['train'],
    'validation': test_eval['test'],
})

splits.save_to_disk("local/data/processed")
