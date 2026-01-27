import os
from huggingface_hub import snapshot_download

# ================= 配置区域 =================
# 1. HuggingFace 上的模型 ID
repo_id = "Qwen/Qwen2.5-Coder-7B-Instruct"

# 2. 你指定的本地保存绝对路径
local_model_path = "/home/mac/PycharmProjects/PythonProject/yoyo/models/Qwen2.5-Coder-7B-Instruct"

# ================= 执行下载 =================
print(f"🚀 正在准备下载模型：{repo_id}")
print(f"📂 保存目标路径：{local_model_path}")

# 确保目录存在
os.makedirs(local_model_path, exist_ok=True)

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_model_path,
        local_dir_use_symlinks=False,  # 【关键点】设为 False，下载的是真实文件，而不是快捷方式
        resume_download=True,          # 支持断点续传，网断了重跑脚本就行
        max_workers=8                  # 开启多线程下载，速度更快
    )
    print("\n✅ 模型下载完成！")
    print(f"请在训练脚本中将 model_path 设置为：\n{local_model_path}")

except Exception as e:
    print(f"\n❌ 下载出错: {e}")
    print("建议检查网络，或者尝试开启 VPN/代理。")