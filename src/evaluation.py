import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from config import PRETRAIN_MODEL_NAME
from datasets import Dataset
import pandas as pd
import numpy as np

import json
import re
from datetime import datetime


role = """
You are a clinical decision model. Provide exactly ONE answer. Return a single JSON object using this format.
{
  "answer": "Yes" or "No",
  "dosage": "Low", "High", or "Omitted",
  "rationale": "[Brief explanation, in 2 sentence]"
}
"""

# query_prompt = """
# Would you offer {drug} to Patient A?  
# Please respond in the following JSON format:  
# ```json
# {{
#   "answer": "Yes" or "No",
#   "dosage": "Low", "High", or "Omitted",
#   "rationale": "[Brief explanation, in 2 sentence]"
# }}
# ```
# """

query_prompt = """
Would you offer {drug} to Patient A?
"""



def evaluate(model_name, tokenizer, model, dataset, max_length=128):
    model.eval()
    scores = []
    records = []
    for sample in tqdm(dataset, desc=f"Evaluating {model_name}"):
        # input_text = sample["instruction"] + " " + sample["input"]

        # vignette,pain_type,pain_severity,question_drug,question_dosage,answer_bool,explanation

        input_text = role + sample["vignette"] + " " + query_prompt.format(drug=sample["drug"])        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(**inputs, 
                                    max_new_tokens=max_length,
                                    temperature=0.0, 
                                    do_sample=False,
                                    )
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)

        # Remove the prompt from the generated output to avoid duplication
        if prediction.startswith(input_text):
            prediction = prediction[len(input_text):].strip()
        
        print(f"###############################################################################")
        print(f"Input: {input_text}")        
        print(f"Prediction: {prediction}")

        parsed_prediction = parse_llm_output(prediction)
        score = get_score(sample, parsed_prediction)
        scores.append(score)

        record = { 
            **sample,
            "llm_answer" : parsed_prediction["answer"],
            "llm_dosage" : parsed_prediction["dosage"],
            "llm_rationale": parsed_prediction["rationale"],
            "score" : score,
        }

        records.append(record)
        
    accuracy = np.average(scores)
    print(f"{model_name} Accuracy: {accuracy:.2%}")

    details = pd.DataFrame(records)
    return accuracy, details

def parse_llm_output(llm_output) -> dict:

    result = {
        "answer":"unknown",
        "dosage":"unknown",
        "rationale": "unknown"
    }
    
    # Step 1: Extract JSON block
    match = re.search(r'```json\s*({.*)', llm_output, re.DOTALL)
    if not match:
        match = re.search(r'({.*)', llm_output, re.DOTALL)
    if not match:
        print("No JSON block found.")
        return result

    json_str = match.group(1).strip()
    json_str = re.sub(r'```$', '', json_str.strip())

    # Step 2: Fix unbalanced quotes
    quote_count = json_str.count('"')
    if quote_count % 2 != 0:
        json_str += '"'  # close the final quote

    # Step 3: Fix unbalanced braces
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if close_braces < open_braces:
        json_str += '}' * (open_braces - close_braces)

    # step 4: trim extra information
    json_str = json_str.split("}")[0] + "}"

    # Step 4: Try parsing
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Still invalid JSON: {json_str}")        

    return result

def get_score(reference, prediction):
    
    # 100 full mark, bool 45, dosage 45, rationale: 20
    score = 0
    if "answer" in prediction.keys():
        if reference["answer_bool"].lower() == prediction["answer"].lower():
            score += 50
    else:
        print("answer not in key!!")

    if "dosage" in prediction.keys():        
        if prediction["dosage"].lower() in reference["dosage"].lower():
            score += 50
    else:
        print("answer not in key!!")

    # TODO score against similarity

    return score/100








def main():
    # Load test data (replace with your medical dataset)
    # dataset = load_dataset("local/data/evaluate/chatgpt_generated.jsonl")
    # dataset = load_from_disk("local/data/processed")["validation"]

    df = pd.read_csv("local/data/evaluation.csv")[:20]
        
    dataset = Dataset.from_pandas(df)

    # Load base model (before fine-tuning)
    base_model_name = PRETRAIN_MODEL_NAME

    # base_model_name = 'local/model/qlora-output'

    
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)    
    # Load quantized model with 4-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        # use_safetensors=True,
        device_map={"": 0},   # force model to GPU device 0
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        },
    )


    print(f"Evaluating model [{base_model_name}] ...")
    average_score, details = evaluate("Base Model", base_tokenizer, base_model, dataset)

    formatted_formal_name = re.sub(r'[^a-zA-Z0-9]', '_', base_model_name)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    details.to_csv(f"eval_{formatted_formal_name}_score_{average_score:.2f}_ts_{ts}.csv")

if __name__ == "__main__":
    main()