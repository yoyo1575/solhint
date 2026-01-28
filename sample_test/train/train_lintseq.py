import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForLanguageModeling
import numpy as np

# --- 1. 核心路径与超参调整 ---
MODEL_ID = "/home/mac/PycharmProjects/PythonProject/yoyo/models/Qwen2.5-Coder-7B-Instruct"
DATA_PATH = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/data/train_lintseq.jsonl"
OUTPUT_DIR = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/lora/solidity_lintseq"

MAX_SEQ_LENGTH = 2048
# 5090 D 显存大，可以将 batch 调至 8-16
BATCH_SIZE = 8           
# 梯度累积，有效 Batch Size = BATCH_SIZE * GRAD_ACCUMULATION = 32
GRAD_ACCUMULATION = 4    
LEARNING_RATE = 1e-4    
NUM_EPOCHS = 3

# --- 2. 健壮的数据整理器 ---
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(self, response_template, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template
        self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

    def __call__(self, examples):
        batch = super().__call__(examples)
        labels = batch["labels"].clone()
        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in range(len(labels[i]) - len(self.response_token_ids)):
                if np.all(labels[i][idx: idx + len(self.response_token_ids)].cpu().numpy() == self.response_token_ids):
                    response_token_ids_start_idx = idx
                    break
            if response_token_ids_start_idx is None:
                labels[i, :] = -100
            else:
                response_start = response_token_ids_start_idx + len(self.response_token_ids)
                labels[i, :response_start] = -100
        batch["labels"] = labels
        return batch

def main():
    # --- 3. 预处理优化 ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def tokenize_function(example):
        instruction = example['instruction']
        input_text = example['input'] if example['input'] else ""
        output = example['output']
        user_content = f"{instruction}\n\nInput:\n{input_text}" if input_text.strip() else instruction
        messages = [{"role": "user", "content": user_content}, {"role": "assistant", "content": output}]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH)

    # 10w条数据开启多进程处理，提升加载速度
    dataset = raw_dataset.map(
        tokenize_function, 
        remove_columns=raw_dataset.column_names,
        num_proc=8,  # 根据 CPU 核心数调整
        load_from_cache_file=True
    )

    # --- 4. 模型加载 ---
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16, 
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True
    )

    peft_config = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)

    # 
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        bf16=True, 
        logging_steps=50,       # 日志频率
        
        # 核心修改：保存策略调整
        save_strategy="steps",
        save_steps=1000,        # 每 1000 步保存一次
        save_total_limit=3,     # 最多保留 3 个模型
        
        optim="adamw_torch",
        gradient_checkpointing=True,
        lr_scheduler_type="cosine", # 余弦退火学习率更适合大数据
        warmup_ratio=0.03,      # 增加预热比例
        report_to="none",
    )

    collator = DataCollatorForCompletionOnlyLM("<|im_start|>assistant\n", tokenizer=tokenizer)

    # 直接使用原生 Trainer，彻底避开 trl 的 Bug
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=collator,
        args=training_args,
    )

    print(f"--- 开始训练 ---")
    trainer.train()
    
    # 最终保存
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
    #8
