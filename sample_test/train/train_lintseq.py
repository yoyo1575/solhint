import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,  # 直接使用基类 Trainer 绕过 trl 的 Bug
)
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForLanguageModeling
import numpy as np

# 1. 配置
MODEL_ID = "/home/mac/PycharmProjects/PythonProject/yoyo/models/Qwen2.5-Coder-7B-Instruct"
DATA_PATH = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/data/train_lintseq.jsonl"
OUTPUT_DIR = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/lora/solidity_lintseq"

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

# 2. 数据整理器 (保持你的逻辑)
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
    # 3. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. 手动预处理：将数据转化为输入 ID
    raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    def tokenize_function(example):
        instruction = example['instruction']
        input_text = example['input'] if example['input'] else ""
        output = example['output']
        user_content = f"{instruction}\n\nInput:\n{input_text}" if input_text.strip() else instruction
        messages = [{"role": "user", "content": user_content}, {"role": "assistant", "content": output}]
        
        # 应用模板并直接分词
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH)

    # 预处理数据并删除原始列
    dataset = raw_dataset.map(tokenize_function, remove_columns=raw_dataset.column_names)

    # 5. 加载模型 (针对 5090 D 优化)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,  # 解决过时警告
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True
    )

    # 6. 配置 LoRA 并应用到模型
    peft_config = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 7. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        bf16=True, # 5090 D 核心优势
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
    )

    # 8. 初始化原始 Trainer (彻底避开 tokenizer 报错)
    collator = DataCollatorForCompletionOnlyLM("<|im_start|>assistant\n", tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        data_collator=collator,
        args=training_args,
    )

    print("--- 5090 D 环境就绪，开始训练 ---")
    trainer.train()
    
    # 9. 保存
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
    #7
