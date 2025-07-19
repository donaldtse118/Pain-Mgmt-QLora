import ast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

from config import PRETRAIN_MODEL_NAME
from datasets import Dataset


from data import prompt

import pandas as pd
import numpy as np

import json
import re
from datetime import datetime


def evaluate(model_name, 
             tokenizer, 
             model, 
             dataset, 
            #  max_length=128,
            max_length=512,
             ):
    model.eval()
    scores = []
    records = []
    for sample in tqdm(dataset, desc=f"Evaluating {model_name}"):
        
        # input_text = prompt.role_and_format + sample["vignette"] + " " + prompt.dosage_query.format(drug=sample["drug"])        
        input_text = prompt.generate_llm_input(sample["vignette"], sample["drug"])
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
        
        # print(f"###############################################################################")
        # print(f"Input: {input_text}")        
        # print(f"Prediction: {prediction}")

        parsed_prediction = parse_llm_output(prediction)
        score = get_score(sample, parsed_prediction)
        scores.append(score)

        if score == 0:
            print(f"Prediction: {prediction}")

        record = { 
            **sample,
            "llm_answer" : parsed_prediction["answer"],
            "llm_dosage" : parsed_prediction["dosage"],
            "llm_rationale": parsed_prediction["rationale"],
            "score" : score,
        }

        records.append(record)

        accuracy = np.average(scores)
        print(f"real time accuracy: {accuracy:.2%}")
        
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
        return ast.literal_eval(json_str)        
    except Exception as e:
        try:
            return json.loads(json_str)        
        except Exception as e:
            print(e)

    return result    

def get_score(reference, prediction):
    
    # 100 full mark, bool 45, dosage 45, rationale: 10
    score = 0
    if "answer" in prediction.keys():
        if reference["answer_bool"].lower() == prediction["answer"].lower():
            score += 50
    else:
        print("answer not in key!!")

    if "dosage" in prediction.keys():        
        pred_dosage = prediction["dosage"].split(" ")[0].lower() # normalise "Low" vs "Low (10mg)" vs "Low (1week)"
        if pred_dosage in reference["dosage"].lower():
            score += 50
    else:
        print("answer not in key!!")

    # TODO score against similarity

    return score/100








def main():
        
    test_datasets = {        
        "augmented" : load_from_disk("local/data/augmented_extend_pain_type_desc")["test"].select(range(20)),        
        "medicial": load_from_disk("local/data/processed")["test"].select(range(20)),
    }

    models = {
        "baseline": PRETRAIN_MODEL_NAME,
        "finetune" : 'local/model/qlora_ft_lmsys_vicuna_7b_v1_5',
    }

    result = []
    
    for model_type, model_name_path in models.items():

        print(f"loaded {model_type} model: [{model_name_path}]")

        tokenizer = AutoTokenizer.from_pretrained(model_name_path)    
        # Load quantized model with 4-bit
        model = AutoModelForCausalLM.from_pretrained(
            model_name_path,
            device_map={"": 0},   # force model to GPU device 0        
            quantization_config={
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            },
        )
        
        for dataset_name, dataset in test_datasets.items():

            print(f"loaded {dataset_name} data ({len(dataset)}) records")
            print(f"Evaluating {model_type} model [{model_name_path}] ...")
            average_score, details = evaluate("Base Model", tokenizer, model, dataset)

            # formatted_model_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name_path.split("/")[-1])
            formatted_model_name = model_type
            
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            details.to_csv(f"eval_{dataset_name}_by_{formatted_model_name}_score_{average_score:.2f}_ts_{ts}.csv")

            result.append({"model_type":model_type,
                           "test_dataset":dataset_name,
                           "score":average_score
                           })
            
        
        # Delete the model explicitly
        del model        
        del tokenizer
        
        # Empty PyTorch CUDA cache
        torch.cuda.empty_cache()
                    
    print(pd.DataFrame(result))    

if __name__ == "__main__":
    main()