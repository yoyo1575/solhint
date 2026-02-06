import torch
import json
import re
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# ================= 配置 =================
BASE_MODEL = "/home/mac/PycharmProjects/PythonProject/yoyo/models/Qwen2.5-Coder-7B-Instruct"
LORA_PATH = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/lora/solidity_lintseq"
OUTPUT_FILE = "solutions_with_diff.json"
# =======================================

def parse_lintseq_diff(raw_output):
    """
    【指标 4 核心逻辑】: 解析 Diff 格式
    返回: (is_valid_diff, clean_code)
    """
    lines = raw_output.split('\n')
    clean_lines = []
    has_diff_header = False
    valid_changes = False

    try:
        for line in lines:
            # 检查是否有 Diff 头 (@@ ... @@)
            if line.startswith('@@') and line.endswith('@@'):
                has_diff_header = True
                continue
            
            # 提取新增行
            if line.startswith('+'):
                clean_lines.append(line[1:]) # 去掉 +
                valid_changes = True
            elif line.startswith('-'):
                continue # 忽略删除行
            else:
                # 某些模型可能混杂纯文本，如果没+号但也不是-号，保留
                clean_lines.append(line)
        
        cleaned_code = '\n'.join(clean_lines)
        
        # 判定标准：只要包含 Diff 头或者有 + 号修改，就算格式合法
        is_valid = has_diff_header or valid_changes
        
        # 兜底：如果清洗出来是空的，说明解析失败
        if not cleaned_code.strip():
            is_valid = False
            
        return is_valid, cleaned_code

    except Exception:
        return False, ""

def main():
    print("🚀 Loading Model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    print("📚 Loading HumanEval-Solidity...")
    dataset = load_dataset("structures-research/HumanEval-Solidity", split="test")
    # dataset = dataset.select(range(10)) # 调试时解开这行，只跑10个

    results = []
    diff_valid_count = 0

    print("⚡ Start Generation...")
    for task in tqdm(dataset):
        prompt = task['prompt']
        
        # 构造输入
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1024, temperature=0.2, top_p=0.95, do_sample=True
            )
        
        raw_output = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        # --- 计算 Metric 4: Diff Validity ---
        is_valid_diff, clean_code = parse_lintseq_diff(raw_output)
        if is_valid_diff:
            diff_valid_count += 1
        
        # 简单的后处理：如果 clean_code 里没有 prompt，把 prompt 拼回去 (为了能编译)
        if "contract " not in clean_code and "function " not in clean_code:
            final_code = prompt + "\n" + clean_code
        else:
            final_code = clean_code

        results.append({
            "task_id": task['task_id'],
            "raw_output": raw_output,
            "final_code": final_code,
            "diff_valid": is_valid_diff,
            "test_code": task['test'] # 用于 Pass@1
        })

    # 保存
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*40)
    print(f" Metric 4: Diff Validity Rate")
    print(f"Valid Diffs: {diff_valid_count}/{len(dataset)}")
    print(f"Score: {diff_valid_count/len(dataset)*100:.2f}%")
    print("="*40)
    print(f"结果已保存到 {OUTPUT_FILE}，请运行 step2_evaluate.py")

if __name__ == "__main__":
    main()
