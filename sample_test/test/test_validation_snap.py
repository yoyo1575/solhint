#!/usr/bin/env python3
import json
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# 隔一级用一个".." 两个用两个 "..",".."
target_dir = os.path.abspath(os.path.join(current_dir, ".."))
if target_dir not in sys.path:
    sys.path.append(target_dir)


from solc_windows import SolidityLinter

def show_code_with_context(code: str, error_lines: list, context: int = 3):
    """与 sample_test 相同的上下文显示函数"""
    lines = code.splitlines()
    error_set = set(error_lines)

    if not lines:
        print("EMPTY")
        return

    to_show = set()
    for ln in error_lines:
        if 1 <= ln <= len(lines):
            start = max(1, ln - context)
            end = min(len(lines), ln + context)
            for k in range(start, end + 1):
                to_show.add(k)

    if not to_show:
        to_show = set(range(1, min(11, len(lines) + 1)))

    for i in range(1, len(lines) + 1):
        if i not in to_show:
            if i == min(to_show) - 1:
                print("    ...")
            continue

        prefix = f"{i:4d}: "
        content = lines[i - 1]
        if i in error_set:
            print(prefix + ">> " + content)
        else:
            print(prefix + "   " + content)

def validate_snapshots(json_file: str):
    """读取 JSON 文件并检查每个 valid_snapshot 是否能通过编译器验证"""
    linter = SolidityLinter()

    resul = []

    # 读取 JSON 文件
    with open(json_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"读取 JSON 失败: {e}")
            return

    # 获取 valid_snapshots
    valid_snapshots = data.get("valid_snapshots", [])
    if not valid_snapshots:
        print(f"'valid_snapshots' 字段不存在或为空")
        return

    # print(f"共有 {len(valid_snapshots)} 个 valid_snapshots")

    for snapshot_idx, snapshot_code in enumerate(valid_snapshots, start=1):
        print(f"\n=== Snapshot {snapshot_idx} ===")
        #print(f"Snapshot Code:\n{snapshot_code}\n")

        # 运行 solc，stderr 会在 get_error_lines 内部自动打印
        try:
            # 忽略警告
            error_lines = linter.get_error_lines_only_error_check(snapshot_code)
        except Exception as e:
            print(f"get_error_lines's return: {e}")
            continue

        # 如果有错误，打印错误信息
        if error_lines:
            print(f"error：", sorted(error_lines))
        else:
            print("Benign")

        # 打印代码上下文
        #show_code_with_context(snapshot_code, error_lines)

if __name__ == "__main__":

    json_file = r"D:\paper\Synthtic_edit_sequence\lintseq\solhint\gen_sol_specific\contract_4_old.json"  # 生成的 JSON 文件路径

    # 从 JSON 文件读取并测试每个 valid_snapshot
    validate_snapshots(json_file)
