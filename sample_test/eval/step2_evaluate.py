import json
import os
import subprocess
import re
from solcx import compile_source, install_solc, set_solc_version

# ================= 配置 =================
INPUT_FILE = "solutions_with_diff.json"
ENABLE_PASS_AT_1 = True  # 如果没装 Foundry，改成 False
TEMP_TEST_DIR = "temp_test_env" # 刚才 forge init 的目录
# =======================================

# 准备编译器
try:
    set_solc_version('0.8.20')
except:
    install_solc('0.8.20')
    set_solc_version('0.8.20')

def check_csr(code):
    """指标 1: 编译通过率"""
    try:
        # 尝试直接编译
        compile_source(code)
        return True
    except:
        # 尝试包裹 contract 再编译
        try:
            wrapped = f"// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\ncontract Test {{\n{code}\n}}"
            compile_source(wrapped)
            return True
        except:
            return False

def check_security(code):
    """指标 2: 安全性 (Slither)"""
    filename = "temp_security.sol"
    with open(filename, "w") as f:
        # Slither 需要完整的 pragma 和 contract
        if "contract " not in code:
            f.write(f"// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\ncontract Test {{\n{code}\n}}")
        else:
            f.write(code)
    
    try:
        # 运行 Slither，只关心 High/Medium 漏洞
        # Slither 返回非 0 表示发现问题或运行错误
        cmd = ["slither", filename, "--json", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                if not data['success']: return False # 运行失败视为不安全
                
                high_bugs = [d for d in data['results']['detectors'] if d['impact'] == 'High']
                return len(high_bugs) == 0 # 没有 High 漏洞才算通过
            except:
                return False
        return True # 如果没输出通常意味着没问题
    except:
        return False

def check_pass_at_1(task_id, code, test_code):
    """指标 3: Pass@1 (Foundry)"""
    if not ENABLE_PASS_AT_1: return False

    # 1. 构造测试文件内容
    # HumanEval 的 Test Code 通常是一个完整的 Contract，我们需要把生成的 Code 塞进去或者引用它
    # 这里采用简单的拼接策略，视 HumanEval-Solidity 的具体格式而定
    # 假设 test_code 依赖名为 "Solution" 的合约
    
    full_contract = f"""
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

{code}

{test_code}
"""
    test_file_path = os.path.join(TEMP_TEST_DIR, "test", f"Test_{task_id}.t.sol")
    
    with open(test_file_path, "w") as f:
        f.write(full_contract)
    
    # 2. 运行 forge test
    # --match-path 指定只跑这个文件
    cmd = ["forge", "test", "--match-path", test_file_path, "--json"]
    try:
        result = subprocess.run(cmd, cwd=TEMP_TEST_DIR, capture_output=True, text=True)
        if '"submodules":' in result.stdout: # 简单的 JSON 检查
             # 只要没有 "failures" 或者 failures 为 0
             return "FAIL" not in result.stdout and "1 failed" not in result.stdout
        return False
    except:
        return False

def main():
    print(f"读取 {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    metrics = {
        "csr": 0,
        "security": 0,
        "pass1": 0
    }
    total = len(data)

    print(f"🚀 开始评估 {total} 个样本...")
    
    for i, item in enumerate(data):
        code = item['final_code']
        
        # 1. 测 CSR (前提)
        is_compiled = check_csr(code)
        if is_compiled:
            metrics["csr"] += 1
            
            # 2. 只有编译通过了，才测 Security
            if check_security(code):
                metrics["security"] += 1
            
            # 3. 只有编译通过了，才测 Pass@1
            # 注意：这步比较慢
            if check_pass_at_1(item['task_id'], code, item['test_code']):
                metrics["pass1"] += 1
        
        if (i+1) % 10 == 0:
            print(f"已处理: {i+1}/{total}")

    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT")
    print("="*50)
    print(f"Total Samples: {total}")
    print(f"CSR (Compile Rate):       {metrics['csr']/total*100:.2f}%")
    print(f"Security Rate (No High):  {metrics['security']/metrics['csr']*100:.2f}% (of compiled)")
    print(f"Pass@1 (Functional):      {metrics['pass1']/total*100:.2f}%")
    print(f"4️Diff Validity:            (See Step 1 Output)")
    print("="*50)

if __name__ == "__main__":
    main()
