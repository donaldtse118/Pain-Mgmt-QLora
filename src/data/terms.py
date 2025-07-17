# Sample dimensions for augmentation
drugs = [
    "hydrocodone", 
    "intravenous hydromorphone",
    "morphine",     
    "oxycodone"
]


pain_types = [
    "acute_cancer", 
    "acute_non_cancer",
    "chronic_cancer", 
    "chronic_non_cancer",
    "post_op"
]

# from sample data "yes" cases
pain_types_drug_map = {
    "acute_cancer": "intravenous hydromorphone",
    "acute_non_cancer": "intravenous hydromorphone",
    "chronic_cancer": "morphine",
    "chronic_non_cancer": "hydrocodone",
    "post_op": "oxycodone",
}

answers = ["Yes", "No"]
dosages = {
    "hydrocodone": {
        "moderate" :"Low (1 week)",
        "severe": "High (4 weeks)",
    },
    "intravenous hydromorphone": {
        "moderate" :"Low (0.5 mg)",
        "severe": "High (1 mg)",
    },    
    "morphine": {
        "moderate" :"Low (1 week)",
        "severe": "High (4 weeks)",
    },
    "oxycodone": {
        "moderate" :"Low (1 week)",
        "severe": "High (4 weeks)",
    },
}

# Basic explanation templates
explanation_templates = {
    "Yes_Low": "Patient is in significant discomfort but not in extreme pain. A low dose of {drug} is appropriate while further evaluation is underway.",
    "Yes_High": "Patient has severe pain with inadequate relief from other medications. A high dose of {drug} is justified for adequate pain control.",
    "No": "Patient's symptoms are not severe enough to warrant {drug} treatment at this time. Consider alternative therapies.",
    # "No_Overdose": "Patient's pain is moderate, and a high dose of {drug} would be excessive and potentially harmful.",
    # "No_Underdose": "Patient's pain is severe, and a low dose of {drug} would not provide adequate relief."
}

# Additional pain severity mapping for explanations
pain_severity_map = {
    "Low (0.5 mg)": "moderate",
    "Low (1 week)": "moderate",
    "High (1 mg)": "severe",
    "High (4 weeks)": "severe",
    # "N/A": "mild"
}

# Pain severity phrases to include in synthetic vignettes

severities = ["mild", "moderate", "severe"]

severity_phrases = {
    "mild": [
        "reports mild discomfort",
        "pain started recently and is tolerable",
        "can sleep despite the pain",
        "NSAIDs provided some relief"
    ],
    "moderate": [
        "reports persistent pain",
        "has difficulty walking due to pain",
        "pain worsens with movement",
        "ibuprofen has limited effect"
    ],
    "severe": [
        "moaning in pain",
        "unable to sleep due to severe pain",
        "pain is sharp and radiating",
        "no relief from any oral analgesics"
    ]
}

pain_templates = [
    "Patient A with history of {pain_type}",
    "Patient A who presents to your clinic with {pain_type}",
    "Patient A who is {pain_type}",
    "Patient A reports {pain_type}",
    "Patient A is experiencing {pain_type}",
    "Patient A complains of {pain_type}",
    "Patient A suffers from {pain_type}",        
    "Patient A came to the emergency department due to {pain_type}",
    "Patient A presents today with a complaint of {pain_type}",
    "Patient A returns for follow-up on {pain_type}",
]

severity_templates = [    
    "Patient A {phrase}",
    "The patient {phrase} recently",
    "There is {phrase} in Patient A",
    "Currently, the patient {phrase}",
    "Right now, the patient {phrase}",
    "There is evidence that the patient {phrase}",
    "The document indicate that the patient {phrase}",
    "It is noted that the patient {phrase}",
    "Clinically, patient A {phrase}",
    "A report indicates that Patient A {phrase}"
]

diagnosis_templates = [
    "Pain is categorized as {severity}",
    "The pain level is assessed as {severity}",
    "Patient is experiencing {severity} pain",
    "Clinical assessment indicates {severity} pain",
    "Pain severity is documented as {severity}",
    "The patient describes the pain as {severity}",
    "Severity of pain is rated {severity}",
    "Pain has been classified as {severity}",
    "Reported pain intensity is {severity}",
    "Evaluation shows signs of {severity} pain",
]