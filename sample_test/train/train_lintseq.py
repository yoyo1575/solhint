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

# 自定义 DataCollator
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(self, response_template, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template
        self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

    def torch_call(self, examples):
        # 兼容旧版本调用方式
        batch = super().torch_call(examples)
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

# 路径配置
MODEL_ID = "/home/mac/PycharmProjects/PythonProject/yoyo/models/Qwen2.5-Coder-7B-Instruct"
DATA_PATH = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/data/train_lintseq.jsonl"
OUTPUT_DIR = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/lora/solidity_lintseq"

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

def main():
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # 强制设置 padding_side
    tokenizer.padding_side = "right"

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            instruction = example['instruction'][i]
            input_text = example['input'][i] if example['input'][i] else ""
            output = example['output'][i]
            user_content = f"{instruction}\n\nInput:\n{input_text}" if input_text.strip() else instruction
            messages = [{"role": "user", "content": user_content}, {"role": "assistant", "content": output}]
            # 使用 apply_chat_template 生成最终文本
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        # 注意：图片提示 torch_dtype 已过时，改用 dtype
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

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        # 显式关闭某些可能导致冲突的设置
        remove_unused_columns=False 
    )

    # 这里的 tokenizer 已经传给 collator 了
    collator = DataCollatorForCompletionOnlyLM("<|im_start|>assistant\n", tokenizer=tokenizer)

    # 关键修改：针对 0.8.6 版本，移除 SFTTrainer 括号内的 tokenizer 参数
    # 因为自定义 collator 内部已有 tokenizer，重复传递会导致 Trainer 初始化异常
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("Starting training on RTX 5090 D...")
    trainer.train()
    
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
    #5
