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
        "reports mild discomfort that does not significantly affect daily activities and remains manageable.",
        "pain started recently and is tolerable, allowing the patient to continue most normal tasks without issue.",
        "can sleep despite the pain, experiencing only occasional interruptions during the night.",
        "NSAIDs provided some relief, though mild soreness persists throughout the day."
    ],
    "moderate": [
        "reports persistent pain that intermittently limits mobility and requires regular attention.",
        "has difficulty walking due to pain, needing to rest frequently to avoid worsening symptoms.",
        "pain worsens with movement, especially during activities that involve bending or lifting.",
        "ibuprofen has limited effect, necessitating the use of additional pain management strategies."
    ],
    "severe": [
        "moaning in pain, visibly distressed and unable to perform even simple tasks.",
        "unable to sleep due to severe pain, with discomfort intensifying at night and causing exhaustion.",
        "pain is sharp and radiating, spreading beyond the initial area and significantly impacting function.",
        "no relief from any oral analgesics, requiring stronger interventions to manage symptoms."
    ]
}


pain_type_desc = {
    'acute_cancer':[
        "abrupt back pain that started while performing normal daily activities; MRI imaging identifies a pathological fracture at L3 with associated canal narrowing and signs of metastatic involvement.",    
        "newly developed midline spinal discomfort; CT imaging reveals widespread lytic lesions and several vertebral fractures, indicative of skeletal metastases.",    
        "acute right upper quadrant pain; abdominal CT identifies a hepatic lesion with internal bleeding.",    
        "sudden back pain following an episode of heavy lifting; MRI demonstrates metastatic lesions in the thoracic spine, with a collapsed vertebral body at T6.",    
        "escalating shoulder pain and numbness in both arms over the past three days; spinal MRI reveals a cystic lesion compressing the spinal cord between C5 and C7, with syrinx formation.",            
    ],
    'acute_non_cancer' : [        
        "sudden onset of severe left lower quadrant abdominal pain following a heavy meal; abdominal CT reveals multiple colonic diverticula with thickening of the sigmoid colon and surrounding fat stranding, consistent with acute diverticulitis.",    
        "severe right leg pain radiating from the lower back to the foot after a gym session; lumbar spine MRI shows a herniated disc at L5-S1 impinging on the right L5 nerve root causing paresthesias and difficulty ambulating.",    
        "progressive, severe periumbilical abdominal pain beginning after the last meal; physical exam reveals rebound tenderness over McBurney’s point and fever, suspicious for acute appendicitis.",    
        "new onset of one of the worst headaches in life associated with visual aura and photosensitivity; head CT and lumbar puncture are negative for hemorrhage, but the patient remains in severe pain requiring urgent management.",       
        "intermittent severe left flank pain radiating to the groin with positive hematuria; abdominal CT identifies multiple stones at the left ureteropelvic junction correlating with recurrent renal colic episodes."
    ],    
    'chronic_cancer':[
        "metastatic breast cancer treated with chemotherapy and hormonal therapy. She now presents with sudden back pain following heavy lifting; MRI shows metastatic lesions in the thoracic spine with a collapsed vertebra at T6.",
        "multiple myeloma and has been stable on maintenance therapy. Recently, he reports newly developed midline spinal discomfort; CT reveals widespread lytic lesions and several vertebral fractures indicating skeletal metastases.",
        "long-standing prostate cancer with bone involvement managed by androgen deprivation therapy. Complains of abrupt back pain that began during normal daily activities; MRI shows a pathological fracture at L3 with canal narrowing and metastatic disease.",
        "chronic hepatitis B and liver cirrhosis with stable liver lesions on surveillance. Presents with right upper quadrant pain; abdominal CT shows a hepatic lesion with internal bleeding.",
        "thyroid cancer treated with radioiodine and regular follow-up. Reports sharp thoracic pain starting three days ago; CT scan reveals bone metastases affecting the sternum and bilateral rib cage."
    ],
    'chronic_non_cancer':[        
        "chronic low back pain attributed to lumbar spondylosis diagnosed several months ago. The pain has progressively worsened despite NSAIDs and physical therapy, with noticeable limitations in daily activities.",
        "persistent right lower back and posterior thigh pain for over two years following a prior injury. MRI confirms a herniated disc at L4-5, and conservative treatments including ibuprofen and physical therapy have offered minimal relief.",
        "ongoing axial low-back pain worsened by bending and lifting, present for the last three months. Despite NSAIDs and recommended exercises, symptoms have progressed, and MRI shows evidence of discitis-osteomyelitis with associated psoas abscesses.",
        "from six months of progressive shoulder pain and hand weakness following a car accident. Imaging reveals a herniated cervical disc causing neuroforaminal and canal stenosis; steroid injections provided only temporary relief.",    
        "with chronic low back and left leg pain after heavy physical activity. Physical exam shows left foot drop and decreased sensation; MRI reveals an L5-S1 disc herniation with canal and neuroforaminal stenosis."
    ],
    'post_op':[
        "post-op day 4 after cervical spine surgery. The patient reports improved arm symptoms but continues to have surgical site pain and residual arm dysesthesias.",    
        "Post-op day 4 following lumbar spine surgery with resolution of prior leg symptoms. The patient has tenderness at the surgical site.",    
        "post-op day 3 after endoscopic nasal surgery. Neurologically intact with expected sanguinous nasal discharge and complains of head pain.",    
        "Post-op day 4 status-post posterior cranial surgery. The patient experienced initial nausea and vomiting but now reports improvement; surgical site pain persists with tenderness to palpation.",    
        "post-op day 16 after cranial surgery. The post-op course is smooth with no complications; residual blood products seen on imaging but neurologically intact."
    ]
}

pain_templates = [
    "Patient A presents with {pain_type}",
    "Patient A reports {pain_type}",
    "Patient A complains of {pain_type}",
    "Patient A has {pain_type}",
    "Patient A is experiencing {pain_type}",
    "Patient A came to the emergency department due to {pain_type}",
    "Patient A presents today with {pain_type}",
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