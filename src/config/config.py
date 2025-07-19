# PRETRAIN_MODEL_NAME = "openchat/openchat-3.5-0106"  # it is too clever, Hard to improve further with LoRA unless data is very domain-specific or changes style drastically.
# PRETRAIN_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Very lightweight, fast to fine-tune <-- but too dumb, cannot return json
# PRETRAIN_MODEL_NAME = "TinyLlama/TinyLlama_v1.1_math_code" # <-- it return code ...
PRETRAIN_MODEL_NAME = "lmsys/vicuna-7b-v1.5" # Lightweight, good for experimentation, open weights <-- not follow instruction well, like json output, make decision
# PRETRAIN_MODEL_NAME = "openchat/openchat_v3.1" #<-- got 13B params, very slow
# PRETRAIN_MODEL_NAME = "openchat/openchat_v3.2"  #<-- got 13B params, very slow