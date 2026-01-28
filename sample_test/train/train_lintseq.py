import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

MODEL_ID = "/home/mac/PycharmProjects/PythonProject/yoyo/models/Qwen2.5-Coder-7B-Instruct"

# 数据路径
DATA_PATH = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/data/train_lintseq.jsonl"

# 输出目录
OUTPUT_DIR = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/lora/solidity_lintseq"

# 显卡优化参数 (针对 5090D)
MAX_SEQ_LENGTH = 2048  # Solidity 代码较长，建议 2048 或 4096
BATCH_SIZE = 8  # 单卡 BS，显存如果够大可以尝试改到 16
GRAD_ACCUMULATION = 2  # 梯度累积，等效 Batch Size = 8 * 2 = 16
LEARNING_RATE = 2e-4  # LoRA 经典学习率
NUM_EPOCHS = 3  # 训练轮数


def main():
    # 1. 加载数据集
    print(f"Loading data from {DATA_PATH}...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. 定义数据格式化函数
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            instruction = example['instruction'][i]
            input_text = example['input'][i]
            output = example['output'][i]

            # 如果 input 不为空，拼接到 instruction 后面
            if input_text and len(input_text.strip()) > 0:
                user_content = f"{instruction}\n\nInput:\n{input_text}"
            else:
                user_content = instruction

            # 构建 Qwen 的 ChatML 对话格式
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]

            # 自动添加 <|im_start|> 等 token
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            output_texts.append(text)
        return output_texts

    # 4. 加载模型 (BF16 + Flash Attention 2)
    print("Loading model with BF16 and Flash Attention 2...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,  # 用 BF16
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )

    # 5. LoRA 配置
    peft_config = LoraConfig(
        r=64,  # LoRA 秩，大一点效果好
        lora_alpha=128,  # alpha 通常是 r 的 2 倍
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # 全模块微调
    )

    # 6. 训练参数设置
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        bf16=True,  # 开启 BF16
        fp16=False,  # 关闭 FP16
        logging_steps=100,  # 每10步打印一次日志

        save_strategy="steps",  # 每轮保存一次
        sava_steps=1000,
        sava_total_limit=3,
        
        optim="adamw_torch",
        report_to="none",  # 不上传 wandb
        gradient_checkpointing=True,  # 显存优化技术，开启后可以跑更大的 Batch
        dataloader_num_workers=4,
    )

    # 7. 定义 DataCollator (关键：只计算回答部分的 Loss)
    # 这让模型只学习“怎么写代码”，不学习“怎么复读问题”
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # 8. 初始化 Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    # 9. 开始训练
    print("🚀 Starting training on RTX 5090D...")
    trainer.train()

    # 10. 保存最终模型
    print(f"✅ Training finished. Saving model to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
