# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
import torch

from config import PRETRAIN_MODEL_NAME

dataset = load_from_disk("local/data/processed")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load quantized model with 4-bit
model = AutoModelForCausalLM.from_pretrained(
    PRETRAIN_MODEL_NAME,    
    # device_map="auto",
    device_map={"": 0},   # force model to GPU device 0
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


def tokenize(example):
    # Concatenate instruction and input as the prompt, output as the label
    prompt = example["instruction"]
    input_text = example.get("input")
    if input_text:
        prompt += "\n" + input_text

    # Tokenize prompt (input) and output (label) separately
    input_ids = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=512,
    )["input_ids"]

    labels = tokenizer(
        example["output"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )["input_ids"]

    return {"input_ids": input_ids, "labels": labels}
tokenized_train_dataset = dataset['train'].map(tokenize, batched=False)

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
    train_dataset=tokenized_train_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Save the LoRA-adapted model and tokenizer
model.save_pretrained("local/model/qlora-output")
tokenizer.save_pretrained("local/model/qlora-output")
