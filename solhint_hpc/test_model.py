import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ================= 配置区域 =================
# 这里的路径必须和你挂载到容器内的路径一致
# 如果你挂载的是 /data，且模型在 /data/models/...
MODEL_PATH = "/workspace/models/Qwen2.5-Coder-7B-Instruct"

# 是否使用 4-bit 量化加载 (与训练环境保持一致)
LOAD_IN_4BIT = True


# ===========================================

def main():
    print(f"检查路径: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型路径！请检查挂载位置。")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"Tokenizer 加载失败: {e}")
        return

    # 配置量化参数 (模拟 QLoRA 训练时的环境)
    quantization_config = None
    if LOAD_IN_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            quantization_config=quantization_config,
            dtype=torch.float16,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ 模型加载失败 (可能是显存不足或 bitsandbytes 问题): {e}")
        return

    print("\n" + "=" * 30)
    print(f"显卡: {torch.cuda.get_device_name(0)}")
    print(f"显存占用: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
    print("=" * 30)
    print("输入 'exit' 或 'quit' 退出\n")

    # === 交互式测试循环 ===
    while True:
        user_input = input("\nPrompt: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input.strip():
            continue

        # 构造对话模板
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": user_input}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)


        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,  # 最大生成长度
                temperature=0.7,  # 随机度
                top_p=0.9
            )

        # 解码并只显示新生成的部分
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print("-" * 20)
        print(response)
        print("-" * 20)


if __name__ == "__main__":
    main()