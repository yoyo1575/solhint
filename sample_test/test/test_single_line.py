#!/usr/bin/env python3
import json
import sys
import os
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


def test_single_line(jsonl_path, line_no):
    """按指定 JSONL 行号测试，输出格式完全模仿 sample_test"""
    linter = SolidityLinter()

    print(f"\n=== Testing JSONL line {line_no} ===")

    # 读取 JSONL 指定行
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if lineno == line_no:
                try:
                    item = json.loads(line)
                except Exception as e:
                    print(f"JSON 解析失败: {e}")
                    return
                break
        else:
            print(f"行号 {line_no} 不存在")
            return

    code = item.get("contract", "")
    if not isinstance(code, str):
        print(f"'contract' 字段不是字符串")
        return

    print(f"Loaded contract at line {line_no}")

    # 运行 solc stderr 会在 get_error_lines 内部自动打印
    try:
        # ignore warning
        error_lines = linter.get_error_lines_only_error_check(code)
    except Exception as e:
        print(f" 编译时出错: {e}")
        return




    # show_code_with_context(code, error_lines)


if __name__ == "__main__":


    jsonl_file = r"D:\paper\Synthtic_edit_sequence\lintseq\solhint\raw_sol\hugging_data_0.jsonl"
    line_no = 1

    test_single_line(jsonl_file, line_no)
