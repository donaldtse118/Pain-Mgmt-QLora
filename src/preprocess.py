from utils import data_loader
def preprocess_data():
    # Load the data
    # raw_data_columns = [Vignette,Question,Answer,Dosage,Explanation]
    df = data_loader.load_raw_data()

    result = []

    

    df.rename(columns={'Vignette': 'instruction',
                       'Question': 'question'},
            inplace=True)
    
    df['Answer'] = df['Answer'].str.replace('Yes.', 'Yes')
    df['Answer'] = df['Answer'].str.replace('No.', 'No')
    df['answer'] = df.apply(lambda r: _format_answer(r), axis=1)

    return df[['instruction','question', 'answer']]

def _format_answer(r):

    short_answer = r['Answer']

    if short_answer == 'Yes':
        short_answer += f", with dosage {r['Dosage']}"

    return f"{short_answer}. {r['Explanation']}"

df = preprocess_data()
df.to_csv('local/data/processed/preprocessed_data.csv', index=False)