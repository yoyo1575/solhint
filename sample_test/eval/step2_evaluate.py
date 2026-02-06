import json
import os
import subprocess
import re
from solcx import compile_source, install_solc, set_solc_version

# ================= 配置 =================
INPUT_FILE = "solutions_with_diff.json"
REPORT_FILE = "final_report.txt"  # 结果保存路径
ENABLE_PASS_AT_1 = True  # 如果没装 Foundry，改成 False
TEMP_TEST_DIR = "temp_test_env" 
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
        compile_source(code)
        return True
    except:
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
        if "contract " not in code:
            f.write(f"// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\ncontract Test {{\n{code}\n}}")
        else:
            f.write(code)
    
    try:
        cmd = ["slither", filename, "--json", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                if not data['success']: return False
                high_bugs = [d for d in data['results']['detectors'] if d['impact'] == 'High']
                return len(high_bugs) == 0 
            except:
                return False
        return True 
    except:
        return False

def check_pass_at_1(task_id, code, test_code):
    """指标 3: Pass@1 (Foundry)"""
    if not ENABLE_PASS_AT_1: return False
    
    full_contract = f"""
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
{code}
{test_code}
"""
    test_file_path = os.path.join(TEMP_TEST_DIR, "test", f"Test_{task_id}.t.sol")
    
    with open(test_file_path, "w") as f:
        f.write(full_contract)
    
    cmd = ["forge", "test", "--match-path", test_file_path, "--json"]
    try:
        result = subprocess.run(cmd, cwd=TEMP_TEST_DIR, capture_output=True, text=True)
        if '"submodules":' in result.stdout:
             return "FAIL" not in result.stdout and "1 failed" not in result.stdout
        return False
    except:
        return False

def main():
    print(f" 读取 {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f" 错误：找不到文件 {INPUT_FILE}，请先运行 step1_generate.py")
        return

    metrics = {
        "csr": 0,
        "security": 0,
        "pass1": 0
    }
    total = len(data)

    print(f"🚀 开始评估 {total} 个样本...")
    
    for i, item in enumerate(data):
        code = item['final_code']
        
        # 1. CSR
        is_compiled = check_csr(code)
        if is_compiled:
            metrics["csr"] += 1
            
            # 2. Security
            if check_security(code):
                metrics["security"] += 1
            
            # 3. Pass@1
            if check_pass_at_1(item['task_id'], code, item['test_code']):
                metrics["pass1"] += 1
        
        if (i+1) % 10 == 0:
            print(f"已处理: {i+1}/{total}")

    # --- 生成报告内容 ---
    csr_rate = metrics['csr'] / total * 100
    sec_rate = (metrics['security'] / metrics['csr'] * 100) if metrics['csr'] > 0 else 0.0
    pass1_rate = metrics['pass1'] / total * 100

    report_lines = [
        "="*50,
        " FINAL EVALUATION REPORT (Solidity LintSeq)",
        "="*50,
        f" Total Samples:          {total}",
        f"  CSR (Compile Rate):     {csr_rate:.2f}%",
        f"  Security Rate (No High): {sec_rate:.2f}% (of compiled)",
        f"  Pass@1 (Functional):     {pass1_rate:.2f}%",
        f"  Diff Validity:           (See step1 log)",
        "="*50,
        "",
        " Detailed Counts:",
        f"   - Compiled:   {metrics['csr']}",
        f"   - Secure:     {metrics['security']}",
        f"   - Passed Test:{metrics['pass1']}",
        "="*50
    ]

    report_content = "\n".join(report_lines)

    # 1. 打印到屏幕
    print("\n" + report_content)

    # 2. 保存到文件
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\n 报告已保存至: {os.path.abspath(REPORT_FILE)}")

if __name__ == "__main__":
    main()
