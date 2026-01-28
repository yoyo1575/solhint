import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling
import numpy as np

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def __init__(self, response_template, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template
        self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

    def torch_call(self, examples):
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

MODEL_ID = "/home/mac/PycharmProjects/PythonProject/yoyo/models/Qwen2.5-Coder-7B-Instruct"
DATA_PATH = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/data/train_lintseq.jsonl"
OUTPUT_DIR = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/lora/solidity_lintseq"

MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

def main():
    print(f"Loading data from {DATA_PATH}...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            instruction = example['instruction'][i]
            input_text = example['input'][i]
            output = example['output'][i]

            if input_text and len(input_text.strip()) > 0:
                user_content = f"{instruction}\n\nInput:\n{input_text}"
            else:
                user_content = instruction

            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            output_texts.append(text)
        return output_texts

    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True
    )

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 修改点：从 SFTConfig 中移除 max_seq_length
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        bf16=True,
        fp16=False,
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=4,
    )

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("Starting training")
    trainer.train()

    print(f"Training finished. Saving model to {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
