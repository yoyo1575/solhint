#!/usr/bin/env python3
import json
import sys
import textwrap

from solc_windows_1 import SolidityLinter

def show_code_with_context(code: str, error_lines: list, context: int = 3):
    lines = code.splitlines()
    error_set = set(error_lines)

    if not lines:
        print("(空代码)")
        return

    to_show = set()
    for ln in error_lines:
        if ln < 1 or ln > len(lines):
            continue
        start = max(1, ln - context)
        end = min(len(lines), ln + context)
        for k in range(start, end + 1):
            to_show.add(k)

    if not to_show:
        to_show = set(range(1, min(11, len(lines) + 1)))

    for i in range(1, len(lines) + 1):
        if i not in to_show:
            if i == min(to_show) - 1:  # 可选：显示省略号
                print("    ...")
            continue
        prefix = f"{i:4d}: "
        content = lines[i-1]
        if i in error_set:
            print(prefix + ">> " + content)
        else:
            print(prefix + "   " + content)


def sample_test(jsonl_file, sample_num=10):
    #  不再传 solc_path，使用多版本自动匹配
    linter = SolidityLinter()

    # print(f"\n开始从 {jsonl_file} 采样 {sample_num} 条合约进行语法检查\n")
    # print(f"可用 solc 版本: {linter.list_available_versions()}")

    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sample_num = min(sample_num, len(lines))

    for idx in range(sample_num):
        jsonl_index = idx + 1
        try:
            item = json.loads(lines[idx])
        except Exception as e:
            print(f"\n=== 样本 #{jsonl_index} JSON 解析失败: {e}")
            continue

        code = item.get("contract", "")
        if not isinstance(code, str):
            print(f"\n=== 样本 #{jsonl_index} 'contract' 字段不是字符串")
            continue
        print(f"sample:{idx+1} and JSONL行号: {jsonl_index}")

        try:
            error_lines = linter.get_error_lines(code)
        except Exception as e:
            print(f" 编译时出错: {e}")
            continue

        if error_lines:
            print("错误行号：", sorted(error_lines))
            # print("\n源码上下文（>> 标记错误行）：\n")
            # show_code_with_context(code, error_lines, context=3)
        else:
            print(" 无语法错误（solc 未报告错误）")

        print("="*60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python sample_test_jsonl_with_context.py <file.jsonl> [sample_n]")
        sys.exit(1)

    jsonl_file = sys.argv[1]
    sample = int(sys.argv[2]) if len(sys.argv) >= 3 else 10
    sample_test(jsonl_file, sample)
