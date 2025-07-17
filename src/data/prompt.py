import json

role_and_format = """
You are a clinical decision model. Provide exactly ONE answer. Return a single JSON object using this format.
{
  "answer": "Yes" or "No",
  "dosage": "Low", "High", or "Omitted",
  "rationale": "[Brief explanation, in 2 sentence]"
}
"""

dosage_query = """
Would you offer {drug} to Patient A?
"""

def generate_llm_input(vignette:str, drug: str):
    return role_and_format + vignette+ " " + dosage_query.format(drug=drug)

def generate_llm_output(answer_bool, dosage, rationale):
    
    result = {
      "answer": answer_bool,
      "dosage": dosage, 
      "rationale": rationale,
    }

    return repr(json.dumps(result)) # repr function adding /n in string