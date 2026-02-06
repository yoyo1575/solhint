import json
from solcx import compile_source, install_solc, set_solc_version

# 安装编译器
try:
    set_solc_version('0.8.20')
except:
    install_solc('0.8.20')
    set_solc_version('0.8.20')

def wrap_and_compile(code):
    """
    尝试编译，如果缺少 contract 包裹则自动添加
    """
    try:
        compile_source(code)
        return True
    except:
        # 尝试包裹一层再编译 (处理只有函数体的情况)
        try:
            wrapped = f"contract Test {{\n{code}\n}}"
            compile_source(wrapped)
            return True
        except:
            return False

def main():
    print("📊 计算 CSR (Compile Success Rate)...")
    with open("solutions.json", "r") as f:
        data = json.load(f)

    total = len(data)
    success = 0

    for item in data:
        if wrap_and_compile(item['solution']):
            success += 1
        else:
            # 可以在这里打印失败的样本ID，方便分析
            pass

    print(f"Total: {total}")
    print(f"Success: {success}")
    print(f"🏆 CSR: {success / total * 100:.2f}%")

if __name__ == "__main__":
    main()
