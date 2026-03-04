import json
import os
import difflib

# ================= 测试配置 =================
# 输入文件：你的原始大文件
INPUT_FILE = r"D:\yoyo\sythtic_edit_sequence\lintseq\solhint\gen_sol_validation_snapshot\gen_sol_HUG_3paths_all.jsonl"

# 输出文件：测试结果
OUTPUT_FILE = r"test_output_fixed.jsonl"

# 测试数量：只跑 5 个看看效果
TEST_LIMIT = 5

# 分隔符
SEPARATOR_TOKEN = "\n<|next_edit|>\n"


# ===========================================

def generate_unified_diff(str_a, str_b):
    """生成简洁的 Unix Diff"""
    # splitlines(keepends=True) 确保保留原有的换行符
    a_lines = str_a.splitlines(keepends=True)
    b_lines = str_b.splitlines(keepends=True)

    diff = difflib.unified_diff(
        a_lines, b_lines,
        fromfile='prev', tofile='curr',
        n=3
    )

    diff_content = []
    for line in diff:
        # 去掉文件头信息，只保留代码变更
        if line.startswith('---') or line.startswith('+++'):
            continue
        diff_content.append(line)

    return "".join(diff_content).strip()


def main():
    print(f"🔍 正在读取前 {TEST_LIMIT} 条数据进行测试 (含换行符修复)...")

    count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
            open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

        for line in f_in:
            if count >= TEST_LIMIT:
                break

            try:
                item = json.loads(line)
                snapshots = item.get("valid_snapshots", [])

                if not snapshots or len(snapshots) < 2:
                    continue

                # 1. 反转列表 (空 -> 完整)
                forward_snapshots = snapshots[::-1]

                edit_sequence_parts = []

                # 2. 计算 Diff
                for i in range(len(forward_snapshots) - 1):
                    prev_state = forward_snapshots[i]
                    curr_state = forward_snapshots[i + 1]

                    # ===============================================
                    # 🛠️【核心修复】强制补全末尾换行符
                    # 避免 difflib 认为 "code" 和 "code\n" 是修改关系
                    # ===============================================
                    if prev_state and not prev_state.endswith('\n'):
                        prev_state += '\n'
                    if curr_state and not curr_state.endswith('\n'):
                        curr_state += '\n'
                    # ===============================================

                    diff_text = generate_unified_diff(prev_state, curr_state)
                    if diff_text:
                        edit_sequence_parts.append(diff_text)

                # 3. 拼接
                full_edit_sequence = SEPARATOR_TOKEN.join(edit_sequence_parts)


                # 5. 写入文件
                output_item = {
                    "original_line_no": item.get("original_line_no"),
                    "edit_sequence": full_edit_sequence
                }
                f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")

                count += 1

            except Exception as e:
                print(f"解析错误: {e}")



if __name__ == "__main__":
    main()