# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

# Load a quantizable model
model_name = "openchat/openchat-3.5-0106"  # or mistralai/Mistral-7B-v0.1 if you have GPU capacity

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load quantized model with 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,    
    device_map="auto",
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
    },
)

# %%
# Prepare model for QLoRA
model = prepare_model_for_kbit_training(model)

# %%
# Configure LoRA
lora_config = LoraConfig(
    r=4,                # Lower rank to reduce #params and overfitting
    lora_alpha=16,      # Lower alpha to soften updates
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.1,   # Slightly higher dropout to prevent overfit
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Sample dataset
dataset = load_dataset("Abirate/english_quotes", split="train[:1000]")  # Small text dataset
def tokenize(example):
    return tokenizer(example['quote'], padding="max_length", truncation=True, max_length=128)
tokenized_dataset = dataset.map(tokenize, batched=True)

# Training config
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=50,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    output_dir="./qlora-output",
    save_total_limit=1,
    save_strategy="no",  # For demo
    report_to="none"
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()
