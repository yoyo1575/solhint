import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ================= ⚙️ HPC 配置区域 =================
MODEL_PATH = "/workspace/models/Qwen2.5-Coder-7B-Instruct"

INPUT_FILE = "/workspace/solhint/data/4_process_merge_3path_diff_data/3path_process_merge_test_data_diff.jsonl"

OUT_DIR = "/workspace/solhint/data/test_data"
OUTPUT_LINTSEQ = os.path.join(OUT_DIR, "test_lintseq.jsonl")
OUTPUT_STANDARD = os.path.join(OUT_DIR, "test_standard.jsonl")

# 测试集建议全量跑，不限制
MAX_UNIQUE_CONTRACTS = 10**9

SYSTEM_PROMPT = """You are an expert Smart Contract Auditor.
Analyze the provided Solidity code and generate a User Instruction.

REQUIREMENTS:
1. **Goal**: Briefly summarize what this function/contract does.
2. **Security**: Explicitly mention specific security patterns used in the code (e.g., "Use SafeMath", "Prevent reentrancy", "Check ownership").
3. **Format**: Output ONLY the instruction text.
"""
# ===================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()

    print(f"Reading input: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")

    # 同一个 original_line_no（3-path）复用 instruction
    instruct_cache = {}

    # 覆盖写，避免断点导致两份输出错位
    f_lint = open(OUTPUT_LINTSEQ, "w", encoding="utf-8")
    f_std = open(OUTPUT_STANDARD, "w", encoding="utf-8")

    llm_calls = 0
    cache_hits = 0
    unique_processed = 0

    for line in tqdm(lines, desc="Generating instructions"):
        try:
            item = json.loads(line)
            idx = item.get("original_line_no")
            final_code = item.get("final_code", "")

            if idx is None or len(final_code) < 20:
                continue

            if idx in instruct_cache:
                instruction = instruct_cache[idx]
                cache_hits += 1
            else:
                if unique_processed >= MAX_UNIQUE_CONTRACTS:
                    break

                code_input = final_code[:4096]
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Code:\n```solidity\n{code_input}\n```\n\nInstruction:"}
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=128,
                        temperature=0.7,
                        do_sample=True
                    )

                generated_ids = [out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated_ids)]
                instruction = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                instruct_cache[idx] = instruction
                llm_calls += 1
                unique_processed += 1

                # 防止缓存过大（可选）
                if len(instruct_cache) > 8000:
                    instruct_cache.clear()
                    instruct_cache[idx] = instruction

            # 实验组：LintSeq（instruction -> edit_sequence）
            lint_sample = {
                "original_line_no": idx,
                "path_id": item.get("path_id"),
                "steps_count": item.get("steps_count"),
                "instruction": instruction,
                "input": "",
                "output": item["edit_sequence"]
            }
            f_lint.write(json.dumps(lint_sample, ensure_ascii=False) + "\n")

            # 对照组：Standard（instruction -> final_code）
            std_sample = {
                "original_line_no": idx,
                "path_id": item.get("path_id"),
                "instruction": instruction,
                "input": "",
                "output": final_code
            }
            f_std.write(json.dumps(std_sample, ensure_ascii=False) + "\n")

        except Exception:
            continue

    f_lint.close()
    f_std.close()

    print("\n" + "=" * 30)
    print("✅ Finished generating test instructions!")
    print(f"Unique contracts: {unique_processed}")
    print(f"LLM calls: {llm_calls}")
    print(f"Cache hits: {cache_hits}")
    print(f"Outputs:")
    print(f"  - {OUTPUT_LINTSEQ}")
    print(f"  - {OUTPUT_STANDARD}")
    print("=" * 30)

if __name__ == "__main__":
    main()