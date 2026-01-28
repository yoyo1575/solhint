import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling
import numpy as np

# 1. 路径配置
MODEL_ID = "/home/mac/PycharmProjects/PythonProject/yoyo/models/Qwen2.5-Coder-7B-Instruct"
DATA_PATH = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/data/train_lintseq.jsonl"
OUTPUT_DIR = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/lora/solidity_lintseq"

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

# 2. 自定义 DataCollator (保持不变)
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(self, response_template, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template
        self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

    def torch_call(self, examples):
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
    # 3. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. 手动预处理数据集（彻底解决 SFTTrainer 内部参数冲突的关键）
    raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    def process_func(example):
        instruction = example['instruction']
        input_text = example['input'] if example['input'] else ""
        output = example['output']
        user_content = f"{instruction}\n\nInput:\n{input_text}" if input_text.strip() else instruction
        messages = [{"role": "user", "content": user_content}, {"role": "assistant", "content": output}]
        # 预先将聊天模板转换为文本
        example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
        return example

    dataset = raw_dataset.map(process_func, remove_columns=raw_dataset.column_names)

    # 5. 加载模型 (针对 5090 D 优化)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16, # 修正警告：使用 dtype 代替 torch_dtype
        attn_implementation="sdpa", # 5090 D 支持
        device_map="auto",
        trust_remote_code=True
    )

    # 6. 配置 LoRA
    peft_config = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 7. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        bf16=True, # 5090 D 必开
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
    )

    collator = DataCollatorForCompletionOnlyLM("<|im_start|>assistant\n", tokenizer=tokenizer)

    # 8. 初始化 SFTTrainer (极简模式)
    # 不再传入 formatting_func 和 tokenizer，直接使用预处理好的 "text" 列
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text", # 指定文本列
        data_collator=collator,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH, # 直接传给 Trainer
    )

    print("Starting training on RTX 5090 D...")
    trainer.train()
    
    # 9. 保存结果
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
    #6
