# solve imbalance class problem

# Combine all splits into a DatasetDict
from datasets import DatasetDict, Dataset

import pandas as pd
from datasets import load_from_disk

exected_dosage = ['high', 'low', 'omitted']

def get_hybrid_train_data(target_size=150):

    df_orig = load_from_disk("local/data/processed")["train"].to_pandas()
    df_aug = load_from_disk("local/data/augmented_extend_pain_type_desc")["train"].to_pandas()

    df_orig['data_source'] = 'orig'
    df_aug['data_source'] = 'augmented'

    df_orig['dosage'] = df_orig['dosage'].apply(lambda x: x.split(" ")[0].lower())
    df_aug['dosage'] = df_aug['dosage'].apply(lambda x: x.split(" ")[0].lower())

    df_aug['pain_type'] = df_aug['pain_type'].str.replace("_", " ")
    
    df_result = df_orig.copy()

    step = 1

    while df_result.shape[0] < target_size:
        # find out least record

        dosage_hist = df_result.dosage.value_counts()

        missed_dosage = [d for d in exected_dosage if d not in dosage_hist]
        if len(missed_dosage) > 0:
            least_dosage = missed_dosage[0]
        else:
            least_dosage = dosage_hist.idxmin()
        least_pain_type = df_result.pain_type.value_counts().idxmin()
        
        # select from augement data
        df_sel = df_aug[(df_aug.pain_type==least_pain_type)& (df_aug.dosage==least_dosage)].head(step)
        df_aug.drop(df_sel.index, inplace=True)

        df_result = pd.concat([df_sel, df_result], ignore_index=True)

    print(df_result.pain_type.value_counts())
    print(df_result.dosage.value_counts())


    return df_result

df = get_hybrid_train_data()
df.to_csv("hybrid_extend_pain_type_desc.csv")


print(f"ttl : {df.shape[0]}")
print(df.groupby(["dosage", "data_source"]).size())

# ttl : 150
# dosage   data_source
# high     augmented      50
# low      augmented      19
#          orig           31
# omitted  augmented      46
#          orig            4

splits = DatasetDict({
    'train': Dataset.from_pandas(get_hybrid_train_data()),
})

splits.save_to_disk("local/data/hybrid_extended_pain_type_desc_200")

