import random
import pandas as pd

from terms import (pain_types, dosages, explanation_templates, severity_phrases, 
                   severity_templates, pain_templates, diagnosis_templates, pain_types_drug_map, severities)



# Generate a synthetic vignette using severity level
def generate_vignette(pain_type, severity):
    
    if severity == "none":
        severity_phrase = random.choice(severity_phrases["mild"])
    else:
        severity_phrase = random.choice(severity_phrases[severity])

    pain_template = random.choice(pain_templates)
    severity_template = random.choice(severity_templates)
    diagnosis_template = random.choice(diagnosis_templates)

    pain_type_str = pain_type.replace('_', ' ')
    
    # return (        
    #     f"{pain_template.format(pain_type=pain_type_str)}. "
    #     f"{severity_template.format(phrase=severity_phrase)}. {diagnosis_template.format(severity=severity)}."
    # )

    return (        
        f"{pain_template.format(pain_type=pain_type_str)}. "
        f"{severity_template.format(phrase=severity_phrase)}."
    )


def generate_augmented_data_with_vignettes(n=50, seed=42):
    random.seed(seed)
    data = []

    for _ in range(n):        
        pain_type = random.choice(pain_types)
        
        drug = pain_types_drug_map[pain_type]        

        patient_severity = random.choice(severities)
        dosage = dosages[drug].get(patient_severity, "Omitted")

        answer = "No" if patient_severity == "mild" else "Yes"

        if answer == "No":
            explanation = explanation_templates[answer].format(drug=drug)
        else:
            if dosage.startswith("Low"):
                explanation = explanation_templates["Yes_Low"].format(drug=drug)
            else:
                explanation = explanation_templates["Yes_High"].format(drug=drug)

        vignette = generate_vignette(pain_type, patient_severity)

        data.append({
            "vignette": vignette,            
            "pain_type": pain_type,
            "pain_severity": patient_severity,
            "drug": drug,
            "dosage": dosage,
            "answer_bool": answer,
            "explanation": explanation
        })

    return pd.DataFrame(data)


# Create a small batch for review
augmented_df = generate_augmented_data_with_vignettes(n=30000)

augmented_df.drop_duplicates(inplace=True)
print(augmented_df.shape)


# augmented_df.to_csv("aug.csv", index=False)
# augmented_df.head(10)

augmented_df = augmented_df.head(1000)

# Prepare dataset
from datasets import Dataset
dataset = Dataset.from_pandas(augmented_df)

# First split: train + temp (test+eval)
train_test = dataset.train_test_split(test_size=0.1, seed=42)  # 70% train, 30% temp

# Second split: test + eval from temp
test_eval = train_test['test'].train_test_split(test_size=0.2, seed=42)  # 15% test, 15% eval

# Combine all splits into a DatasetDict
from datasets import DatasetDict

splits = DatasetDict({
    'train': train_test['train'],
    'validation': test_eval['train'],
    'test': test_eval['test'],
})

splits.save_to_disk("local/data/augmented_no_diagnosis")

