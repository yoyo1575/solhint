import torch
import json
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# ================= 配置路径 =================
BASE_MODEL = "/home/mac/PycharmProjects/PythonProject/yoyo/models/Qwen2.5-Coder-7B-Instruct"
LORA_PATH = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/lora/solidity_lintseq"
OUTPUT_FILE = "solutions.json"

def clean_lintseq_code(text):
    """
    清洗函数：把模型生成的 Diff 格式还原成标准 Solidity 代码
    """
    lines = text.split('\n')
    code_lines = []
    for line in lines:
        # 去掉 diff 头部信息
        if line.startswith('@@') and line.endswith('@@'):
            continue
        # 去掉删除线
        if line.startswith('-'):
            continue
        # 提取新增线 (去掉 + 号)
        if line.startswith('+'):
            code_lines.append(line[1:]) 
        else:
            # 保留原本没有标记的行
            code_lines.append(line)
    return '\n'.join(code_lines)

def main():
    # 1. 加载模型
    print("🚀 Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="sdpa", 
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    # 2. 加载题目集
    print("Loading HumanEval-Solidity...")
    dataset = load_dataset("structures-research/HumanEval-Solidity", split="test")

    # 3. 开始生成
    results = []
    print(f"Start generating for {len(dataset)} tasks...")

    for task in tqdm(dataset):
        task_id = task['task_id']
        prompt = task['prompt'] # 题目描述
        
        # 构造输入
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.2, # 测 Pass@1 建议低温度
                top_p=0.95,
                do_sample=True
            )
        
        # 解码
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
        raw_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 清洗
        clean_code = clean_lintseq_code(raw_output)

        # 拼接：Prompt (函数头) + 生成的代码 (函数体)
        # 注意：有些模型会重复输出 Prompt，这里需要你根据实际情况微调
        # 简单策略：如果生成的不包含 prompt，就拼上去
        full_code = clean_code
        if "contract " not in clean_code and "function " not in clean_code:
             full_code = prompt + "\n" + clean_code

        results.append({
            "task_id": task_id,
            "solution": full_code, 
            "test": task['test'] # 保留测试用例，后面测 Pass@1 要用
        })

    # 保存结果
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ 生成完成，已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
