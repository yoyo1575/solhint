import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig  # 新增：用于量化
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForLanguageModeling
import numpy as np

# --- 自动检测 CPU 核心数 ---
NUM_PROC = max(1, (os.cpu_count() or 4) // 2)

MODEL_ID = "/home/mac/PycharmProjects/PythonProject/yoyo/models/Qwen2.5-Coder-7B-Instruct"
DATA_PATH = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/data/train_lintseq.jsonl"
OUTPUT_DIR = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/lora/solidity_lintseq"

# --- 显存优化参数 (针对 24G 显存) ---
MAX_SEQ_LENGTH = 1024   
BATCH_SIZE = 4          # 5090 D 开启量化后可以尝试 4
GRAD_ACCUMULATION = 8   # 有效 Batch Size = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3

def main():
    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. 10w 数据加速预处理
    raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    def tokenize_func(example):
        instruction = example['instruction']
        input_text = example['input'] if example['input'] else ""
        messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}, 
                    {"role": "assistant", "content": example['output']}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH)

    dataset = raw_dataset.map(tokenize_func, remove_columns=raw_dataset.column_names, num_proc=NUM_PROC)

    # 3. 量化配置：这是解决 OOM 的核心
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 4. 加载模型 (放弃 flash_attn，改用 sdpa)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config, # 开启 4-bit
        dtype=torch.bfloat16,
        attn_implementation="sdpa",     # 5090 D 原生支持，不需要安装额外库
        device_map="auto",
        trust_remote_code=True
    )

    # 5. QLoRA 适配
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)

    # 6. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        bf16=True,
        logging_steps=50,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit", # 进一步压榨显存
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        args=training_args,
    )

    print("--- Flash Attention 停用，改用内置 SDPA。开始训练 ---")
    trainer.train()

if __name__ == "__main__":
    main()
    #9
