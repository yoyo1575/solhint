import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ================= 配置路径 =================
# 1. 基座模型路径 (你的本地路径)
BASE_MODEL_PATH = "/home/mac/PycharmProjects/PythonProject/yoyo/models/Qwen2.5-Coder-7B-Instruct"

# 2. 刚刚训练好的 LoRA 权重路径
LORA_PATH = "/home/mac/PycharmProjects/PythonProject/yoyo/solhint/lora/solidity_lintseq"

# ===========================================

def main():
    print("🚀 正在加载基座模型...")
    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    # 2. 加载基座模型 (使用 BF16 和 SDPA 加速)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa", # 推理时也可以用 sdpa 加速
        trust_remote_code=True
    )

    print(f"🔗 正在挂载 LoRA 权重: {LORA_PATH} ...")
    # 3. 加载并合并 LoRA 权重
    # 这步操作不会修改硬盘上的文件，只是在显存里把 LoRA 贴上去
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    # 切换到评估模式
    model.eval()
    print("✅ 模型加载完毕，准备生成！")

    # ================= 测试案例 =================
    
    # 这里写一个你想测试的 Prompt
    # 注意：这里的 Instruction 风格要和你训练集里的保持一致
    instruction = "Create a standard ERC20 token contract named 'MyToken' with symbol 'MTK'."
    input_text = "" # 如果有 input 就填，没有留空

    # 4. 构造对话格式 (ChatML)
    if input_text:
        content = f"{instruction}\n\nInput:\n{input_text}"
    else:
        content = instruction

    messages = [
        {"role": "user", "content": content}
    ]
    
    # 应用模板
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True # 这一步很关键，告诉模型该轮到 assistant 说话了
    )

    # 5. 编码输入
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 6. 生成代码
    print("\n🤖 正在生成回复...\n" + "="*50)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,   # 生成的最大长度
            temperature=0.2,       # 温度低一点，代码生成的逻辑更严谨
            top_p=0.9,
            do_sample=True
        )

    # 7. 解码输出 (去掉输入的 Prompt 部分，只看新生成的)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(response)
    print("="*50)

if __name__ == "__main__":
    main()
