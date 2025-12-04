#!/usr/bin/env python3
import json
import sys
import textwrap
import random

from solc_windows import SolidityLinter

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


def sample_test(jsonl_file, sample_num):
    import random

    linter = SolidityLinter()

    # 载入 JSONL 全部行
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)

    if total == 0:
        print("JSONL 文件为空")
        return

    sample_num = min(sample_num, total)

    # ★★★ 关键：从所有行中随机抽样 sample_num ★★★
    sampled_indices = random.sample(range(total), sample_num)

    for real_idx, idx in enumerate(sampled_indices, start=1):

        jsonl_index = idx + 1  # 行号（从 1 开始）
        line = lines[idx]

        try:
            item = json.loads(line)
        except Exception as e:
            print(f"\n=== 样本 #{real_idx} (JSONL 行 {jsonl_index}) JSON 解析失败: {e}")
            continue

        code = item.get("contract", "")
        if not isinstance(code, str):
            print(f"\n=== 样本 #{real_idx} JSONL 行 {jsonl_index}: 'contract' 字段不是字符串")
            continue

        print(f"sample {real_idx}/{sample_num}  | JSONL 行号: {jsonl_index}")

        try:
            error_lines = linter.get_error_lines(code)
        except Exception as e:
            print(f" 编译时出错: {e}")
            continue

        if error_lines:
            print("错误行号：", sorted(error_lines))
        else:
            print("BENIGN")

        print("=" * 60 + "\n")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python sample_test_jsonl_with_context.py <file.jsonl> [sample_n]")
        sys.exit(1)

    jsonl_file = sys.argv[1]
    sample = int(sys.argv[2]) if len(sys.argv) >= 3 else 10
    sample_test(jsonl_file, sample)
