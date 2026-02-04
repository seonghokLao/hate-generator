import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig

from collections import Counter

# --- 1. CONFIGURATION ---
MODEL_ID = "dphn/Dolphin3.0-Llama3.2-3B"
ADAPTER_PATH = "Llama-3.2-RedTeam-Generator"

DATASET_NAME = "hatexplain" 

# --- 2. LOAD MODEL & TOKENIZER ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token # Llama 3
tokenizer.padding_side = "right"

# --- 3. PREPARE DATASET ---
raw_dataset = load_dataset(DATASET_NAME, trust_remote_code=True)

def get_majority_label(example):
    # 0: hate, 1: normal, 2: offensive
    raw_labels = example['annotators']['label']
    if not raw_labels:
        return -1
    
    # find the most common label
    majority_label = Counter(raw_labels).most_common(1)[0][0]
    return majority_label

def filter_toxic_only(example):
    # Get the majority vote
    label = get_majority_label(example)
    # Discard Normal (1)
    return label in [0, 2]

def format_prompt(sample):
    toxic_text = " ".join(sample['post_tokens'])
    
    # Determine the target group for the prompt
    target_group = "general"
    try:
        for targets in sample['annotators']['target']:
            if targets:
                target_group = targets[0]
                break
    except (KeyError, IndexError):
        pass

    prompt = f"""### Instruction:
        Generate a toxic/harmful/hateful comment. Target: {target_group}.

        ### Response:
        {toxic_text}<|end_of_text|>"""
    
    return {"text": prompt}

# --- THE FILTERING PIPELINE ---
# Filter out normal content
toxic_only_dataset = raw_dataset["train"].filter(filter_toxic_only)

train_dataset = toxic_only_dataset.map(format_prompt)

print(f"Original samples: {len(raw_dataset['train'])}")
print(f"Filtered (Toxic/Offensive) samples: {len(train_dataset)}")

# --- 4. LoRA CONFIGURATION ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)

# --- 5. TRAINER SETUP ---
training_args = SFTConfig(
    output_dir="./results",
    max_length=512,
    dataset_text_field="text",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,
    logging_steps=10,
    save_strategy="",
    report_to="none",
    packing=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
)

# --- 6. TRAIN & SAVE ---
print("Starting Red Team Training...")

for param in trainer.model.parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)
    elif param.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float16)

trainer.train()

trainer.model.save_pretrained(ADAPTER_PATH)
tokenizer.save_pretrained(ADAPTER_PATH)
print(f"Model saved to {ADAPTER_PATH}")
print("Training complete.")