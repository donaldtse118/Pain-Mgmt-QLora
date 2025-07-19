# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
import torch
import re
from data import prompt

from config import PRETRAIN_MODEL_NAME

# dataset = load_from_disk("local/data/processed")
# dataset = load_from_disk("local/data/augmented")
# dataset = load_from_disk("local/data/augmented_no_diagnosis")
# dataset = load_from_disk("local/data/hybrid")
dataset = load_from_disk("local/data/hybrid_extend_pain_type_desc")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Load quantized model with 4-bit
model = AutoModelForCausalLM.from_pretrained(
    PRETRAIN_MODEL_NAME,    
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
    # r=8,                # Lower rank to reduce #params and overfitting    
    # lora_alpha=32,      # Lower alpha to soften updates
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.1,   # Slightly higher dropout to prevent overfit
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

def tokenize(sample):
    # 1. Generate the full prompt and the expected output
    input_text = prompt.generate_llm_input(sample["vignette"], sample["drug"])
    label_text = prompt.generate_llm_output(
        sample['answer_bool'],
        sample['dosage'],
        sample['explanation']
    )

    # 2. Concatenate input_text and label_text to form the complete sequence for the model
    #    Add an EOS token at the end of the entire sequence if your model expects it
    #    or if you want the generation to naturally stop.
    #    However, for training, just concatenating is often sufficient as the max_length will truncate.
    full_sequence_text = input_text + label_text + tokenizer.eos_token

    # 3. Tokenize the full sequence
    tokenized_full_sequence = tokenizer(
        full_sequence_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    input_ids = tokenized_full_sequence.input_ids[0]
    attention_mask = tokenized_full_sequence.attention_mask[0]

    # 4. Create labels: The labels should be a copy of input_ids,
    #    but with the prompt part masked out.

    # First, determine the length of the tokenized input_text part.
    # This is crucial for correctly masking. Tokenize input_text independently
    # to get its exact token length.
    # Be careful: some tokenizers add an implicit EOS token. You might need to adjust by -1
    # if `tokenizer(input_text)["input_ids"]` ends with EOS and `input_text + label_text`
    # does not result in an extra EOS after input_text.
    tokenized_input_part = tokenizer(input_text, truncation=True)
    input_len = len(tokenized_input_part.input_ids)
    
    # Copy input_ids to create labels
    labels = input_ids.clone() # Use .clone() for tensors

    # Mask out the input_text portion from the labels
    # PyTorch's CrossEntropyLoss with `ignore_index=-100` will ignore these tokens.
    # Ensure you are not masking the first token of the actual output.
    # The -100 masking means the model will not try to predict these tokens.
    labels[:input_len] = -100

    # Replace padding tokens in labels with -100.
    # This is already handled by the `labels[:input_len] = -100` if padding is before the output.
    # If padding is at the end, and part of the output is padded, this line is correct.
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


tokenized_train_dataset = dataset['train'].map(tokenize, batched=False)

# Training config
training_args = TrainingArguments(
    logging_steps=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    # max_steps=100,
    max_steps=60,
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

output = trainer.train()


formatted_model_name = re.sub(r'[^a-zA-Z0-9]', '_', PRETRAIN_MODEL_NAME)

# Save the LoRA-adapted model and tokenizer
output_model_path = f"local/model/qlora_ft_{formatted_model_name}"
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)

print(output.training_loss)
print(output.metrics)


# tokenise input and output together and mask output input
# {'train_runtime': 323.4536, 'train_samples_per_second': 0.989, 'train_steps_per_second': 0.062, 'total_flos': 4974529499627520.0, 'train_loss': 1.5642086505889892, 'epoch': 5.0}
# max_steps=50
# {'train_runtime': 815.4384, 'train_samples_per_second': 0.981, 'train_steps_per_second': 0.061, 'total_flos': 1.258860526436352e+16, 'train_loss': 1.1028675842285156, 'epoch': 12.615384615384615}
# added EOS
# {'train_runtime': 815.8614, 'train_samples_per_second': 0.981, 'train_steps_per_second': 0.061, 'total_flos': 1.258860526436352e+16, 'train_loss': 1.0983434963226317, 'epoch': 12.615384615384615}

# r=8, lora_alpha=32, max_step=50
# {'train_runtime': 817.4, 'train_samples_per_second': 0.979, 'train_steps_per_second': 0.061, 'total_flos': 1.25925995839488e+16, 'train_loss': 0.9513972663879394, 'epoch': 12.615384615384615}
# max_steps=100
# {'train_runtime': 1611.496, 'train_samples_per_second': 0.993, 'train_steps_per_second': 0.062, 'total_flos': 2.4880539500544e+16, 'train_loss': 0.47257256641983986, 'epoch': 25.0}


##### Train with Augmented dataset ###
# {'train_runtime': 419.3328, 'train_samples_per_second': 0.763, 'train_steps_per_second': 0.048, 'total_flos': 6499406236876800.0, 'train_loss': 0.7589183330535889, 'epoch': 0.015063076633402372}
# 900 rcd for training, no max_steps, default 171 
# all correct for valid json (but all yes case)
# {'train_runtime': 29616.9554, 'train_samples_per_second': 0.091, 'train_steps_per_second': 0.006, 'total_flos': 5.4838740123648e+16, 'train_loss': 0.0840140072631749, 'epoch': 3.0}
# {'train_runtime': 418.9467, 'train_samples_per_second': 0.764, 'train_steps_per_second': 0.048, 'total_flos': 6499406236876800.0, 'train_loss': 0.7415579557418823, 'epoch': 0.35555}


# without diagnosis (augmented data: 75%, medical data: 23%)
# {'train_runtime': 419.2248, 'train_samples_per_second': 0.763, 'train_steps_per_second': 0.048, 'total_flos': 6499406236876800.0, 'train_loss': 0.6309847116470337, 'epoch': 0.35555555555555557}

# without diagnosis + r=4, alpha=16, max_steps = 20
# {'train_runtime': 422.6856, 'train_samples_per_second': 0.757, 'train_steps_per_second': 0.047, 'total_flos': 6497344652574720.0, 'train_loss': 0.7833358764648437, 'epoch': 0.35555555555555557}
# model_type  test_dataset  score
# baseline    augmented     0.600
# baseline    medicial      0.775
# finetune    augmented     0.700
# finetune     medicial     0.225

# train with hybrid dataset
# {'train_runtime': 396.5019, 'train_samples_per_second': 0.807, 'train_steps_per_second': 0.05, 'total_flos': 6131869015867392.0, 'train_loss': 1.0309191703796388, 'epoch': 3.3478260869565215}

# train with hybrid dataset(150 rcd), extended pain type and severity description, max_step=20
#   model_type test_dataset  score
# 0   baseline    augmented  0.500
# 1   baseline     medicial  0.775
# 2   finetune    augmented  0.620
# 3   finetune     medicial  0.600
# {'train_runtime': 393.3631, 'train_samples_per_second': 0.813, 'train_steps_per_second': 0.051, 'total_flos': 6091260611788800.0, 'train_loss': 0.96949462890625, 'epoch': 2.0}

# train with hybrid dataset(150 rcd), extended pain type and severity description, max_step=40
#   model_type test_dataset  score
# 0   baseline    augmented  0.500
# 1   baseline     medicial  0.775
# 2   finetune    augmented  0.925
# 3   finetune     medicial  0.700
# {'train_runtime': 787.7507, 'train_samples_per_second': 0.812, 'train_steps_per_second': 0.051, 'total_flos': 1.21825212235776e+16, 'train_loss': 0.6849146366119385, 'epoch': 4.0}

# train with hybrid dataset(150 rcd), extended pain type and severity description, max_step=60
#   model_type test_dataset  score
# 0   baseline    augmented  0.500
# 1   baseline     medicial  0.775
# 2   finetune    augmented  0.925
# 3   finetune     medicial  0.950
# {'train_runtime': 1178.4118, 'train_samples_per_second': 0.815, 'train_steps_per_second': 0.051, 'total_flos': 1.82737818353664e+16, 'train_loss': 0.5354639172554017, 'epoch': 6.0}
