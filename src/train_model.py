# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
import torch
import re
from data import prompt

from config import PRETRAIN_MODEL_NAME

dataset = load_from_disk("local/data/processed")

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
    # r=4,                # Lower rank to reduce #params and overfitting
    # lora_alpha=16,      # Lower alpha to soften updates
    r=8,                # Lower rank to reduce #params and overfitting    
    lora_alpha=32,      # Lower alpha to soften updates
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.1,   # Slightly higher dropout to prevent overfit
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# def tokenize(sample):

#     input_text = prompt.generate_llm_input(sample["vignette"], sample["question_drug"])

#     inputs = tokenizer(
#         input_text,
#         padding="max_length",
#         truncation=True,
#         max_length=512,
#         return_tensors="pt"
#     )
      
#     label_text = prompt.generate_llm_output(sample['answer_bool'],
#                                        sample['answer_dosage'],
#                                        sample['explanation'])

#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(
#             label_text,
#             padding="max_length",
#             truncation=True,
#             max_length=512,
#             return_tensors="pt"
#         )

#     # return {"input_ids": input_ids, "labels": labels}
    
#     input_ids = inputs.input_ids[0]
#     label_ids = labels.input_ids[0]
#     label_ids[label_ids == tokenizer.pad_token_id] = -100  # Replaces all padding tokens in label_ids with -100.-100 is a special value used by PyTorch to ignore that token when computing loss.

#     return {
#         "input_ids": input_ids,
#         "attention_mask": inputs.attention_mask[0],
#         "labels": label_ids,
#     }

def tokenize(sample):
    # 1. Generate the full prompt and the expected output
    input_text = prompt.generate_llm_input(sample["vignette"], sample["question_drug"])
    label_text = prompt.generate_llm_output(
        sample['answer_bool'],
        sample['answer_dosage'],
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
    max_steps=100,
    # max_steps=20,
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



# max_steps=20,
# {'train_runtime': 374.1127, 'train_samples_per_second': 0.855, 'train_steps_per_second': 0.053, 'train_loss': 6.15913200378418, 'epoch': 5.0} 
# {'train_runtime': 319.6118, 'train_samples_per_second': 1.001, 'train_steps_per_second': 0.063, 'train_loss': 6.08679428100586, 'epoch': 5.0}
# max_steps=30
# {'train_runtime': 564.4196, 'train_samples_per_second': 0.85, 'train_steps_per_second': 0.053, 'train_loss': 4.874612426757812, 'epoch': 7.62}
# max_steps=50
# {'train_runtime': 803.9691, 'train_samples_per_second': 0.995, 'train_steps_per_second': 0.062, 'train_loss': 3.45401123046875, 'epoch': 12.62}
# {'train_runtime': 811.0319, 'train_samples_per_second': 0.986, 'train_steps_per_second': 0.062, 'train_loss': 3.4659902954101565, 'epoch': 12.62}

# AFTER replace padding with magic value -100
# max_steps=30
# {'train_runtime': 343.2545, 'train_samples_per_second': 0.932, 'train_steps_per_second': 0.058, 'train_loss': 14.077944946289062, 'epoch': 5.0}
# no max_steps
# {'train_runtime': 195.8288, 'train_samples_per_second': 0.751, 'train_steps_per_second': 0.061, 'total_flos': 2984717699776512.0, 'train_loss': 15.364848136901855, 'epoch': 3.0}

# change rank from 4->8
# {'train_runtime': 194.5023, 'train_samples_per_second': 0.756, 'train_steps_per_second': 0.062, 'total_flos': 2985664740065280.0, 'train_loss': 15.351463317871094, 'epoch': 3.0}

# change lora_alpha from 16->32
# {'train_runtime': 493.2269, 'train_samples_per_second': 0.973, 'train_steps_per_second': 0.061, 'total_flos': 7616491683840000.0, 'train_loss': 11.165579223632813, 'epoch': 7.615384615384615}

# r=4, lora_alpha=16, max_step=50
# {'train_runtime': 814.5736, 'train_samples_per_second': 0.982, 'train_steps_per_second': 0.061, 'total_flos': 1.258860526436352e+16, 'train_loss': 9.612734909057616, 'epoch': 12.615384615384615}

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