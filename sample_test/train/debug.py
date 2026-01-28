import os
import trl

# 1. 找到 trl 安装在哪
trl_path = os.path.dirname(trl.__file__)
print(f"🔍 TRL 安装路径: {trl_path}")

# 2. 遍历所有文件，查找 DataCollatorForCompletionOnlyLM
print("🚀 开始全盘搜索...")
found = False
for root, dirs, files in os.walk(trl_path):
    for file in files:
        if file.endswith(".py"):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "class DataCollatorForCompletionOnlyLM" in content:
                        print(f"\n✅ 找到了！")
                        print(f"📄 文件位置: {file_path}")
                        
                        # 计算导入路径
                        rel_path = os.path.relpath(file_path, os.path.dirname(trl_path))
                        import_path = rel_path.replace("/", ".").replace(".py", "")
                        print(f"💡 你应该这样导入: from {import_path} import DataCollatorForCompletionOnlyLM")
                        found = True
            except:
                pass

if not found:
    print("\n❌ 完蛋，文件里真没有这个类。说明你的安装包是残缺的。")
