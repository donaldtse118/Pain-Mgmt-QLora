import random
import pandas as pd

from terms import drugs, pain_types, answers, dosages, explanation_templates, pain_severity_map, severity_phrases, pain_types_drug_map, severities



# Generate a synthetic vignette using severity level
def generate_vignette(pain_type, severity):
    
    if severity == "none":
        phrase = random.choice(severity_phrases["mild"])
    else:
        phrase = random.choice(severity_phrases[severity])
    
    return (
        f"Patient A with {pain_type.replace('_', ' ')}. "
        f"The patient A {phrase}. Pain is categorized as {severity}."
    )

# def dignose(patient_severity, dosage_target_severity):

#     answer = "Unknown"

#     if patient_severity == dosage_target_severity:
#         answer = "Yes"
#         return answer

#     if patient_severity == "mild":
#         answer = "No_Opioid" # no opioid for mild pain
#     elif patient_severity == "moderate":
#         if dosage_target_severity == "severe":
#             answer = "No_Overdose" # too much dosage for moderate pain
#     elif patient_severity == "severe":
#         if dosage_target_severity == "moderate":
#             answer = "No_Underdose" # not enough dosage for severe pain

#     return answer

def generate_augmented_data_with_vignettes(n=50, seed=42):
    random.seed(seed)
    data = []

    for _ in range(n):        
        pain_type = random.choice(pain_types)
        
        drug = pain_types_drug_map[pain_type]
        
        # question_dosage = random.choice(dosages[question_drug])
        # dosage_target_severity = pain_severity_map.get(question_dosage, "none")

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
augmented_df = generate_augmented_data_with_vignettes(n=100)
augmented_df.to_csv("aug.csv", index=False)
augmented_df.head(10)
