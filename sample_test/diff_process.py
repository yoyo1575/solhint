import json
import os
import difflib
from tqdm import tqdm

# ================= 正式配置 =================
# 输入：LintSeq 原始结果
INPUT_FILE = r"D:\yoyo\sythtic_edit_sequence\lintseq\solhint\gen_sol_validation_snapshot\gen_sol_HUG_3paths_all.jsonl"

# 输出：处理好的 Diff 数据
OUTPUT_FILE = r"D:\yoyo\sythtic_edit_sequence\lintseq\solhint\data\processed_diffs_only.jsonl"

# 分隔符：这里去掉了 <|next_edit|>，只用换行符连接
# 这样生成的格式就像是一个包含了多个补丁的标准 Patch 文件
JOIN_SEPARATOR = "\n"


# ===========================================

def generate_unified_diff(str_a, str_b):
    """生成简洁的 Unix Diff"""
    a_lines = str_a.splitlines(keepends=True)
    b_lines = str_b.splitlines(keepends=True)

    diff = difflib.unified_diff(
        a_lines, b_lines,
        fromfile='prev', tofile='curr',
        n=3  # 上下文保留3行
    )

    diff_content = []
    for line in diff:
        # 去掉文件头信息 (--- prev, +++ curr)，只保留代码变更
        if line.startswith('---') or line.startswith('+++'):
            continue
        diff_content.append(line)

    # strip() 会去掉末尾的换行符，所以我们在拼接时要注意补上
    return "".join(diff_content).strip()


def main():
    # 确保输出目录存在
    out_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"🚀 开始全量处理 Diff 数据 (无特殊Token版)...")
    print(f"输入: {INPUT_FILE}")
    print(f"输出: {OUTPUT_FILE}")

    processed_count = 0
    skipped_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
            open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

        # 读取所有行用于进度条
        lines = f_in.readlines()

        for line in tqdm(lines, desc="Processing"):
            try:
                item = json.loads(line)
                snapshots = item.get("valid_snapshots", [])

                # 数据清洗：如果没有快照或只有一个（没发生变化），跳过
                if not snapshots or len(snapshots) < 2:
                    skipped_count += 1
                    continue

                # 1. 反转列表 (空 -> 完整)
                forward_snapshots = snapshots[::-1]

                edit_sequence_parts = []

                # 2. 计算 Diff (带换行符修复)
                for i in range(len(forward_snapshots) - 1):
                    prev_state = forward_snapshots[i]
                    curr_state = forward_snapshots[i + 1]

                    # === 🛠️ 修复换行符 (Crucial Fix) ===
                    # 强制给没有换行符的代码补全换行，保证 Diff 格式整洁
                    if prev_state and not prev_state.endswith('\n'):
                        prev_state += '\n'
                    if curr_state and not curr_state.endswith('\n'):
                        curr_state += '\n'
                    # ===================================

                    diff_text = generate_unified_diff(prev_state, curr_state)

                    if diff_text:
                        edit_sequence_parts.append(diff_text)

                if not edit_sequence_parts:
                    skipped_count += 1
                    continue

                # 3. 拼接
                # 使用换行符拼接，去掉之前的 <|next_edit|>
                full_edit_sequence = JOIN_SEPARATOR.join(edit_sequence_parts)

                # 4. 保存
                output_item = {
                    "original_line_no": item.get("original_line_no"),
                    "final_code": item.get("final_code") or snapshots[0],
                    "edit_sequence": full_edit_sequence,
                    "steps_count": len(edit_sequence_parts)
                }

                f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                processed_count += 1

            except Exception as e:
                print(f"Error processing line: {e}")
                continue


    print("✅ 全量处理完成！")
    print(f"总数据量: {len(lines)}")
    print(f"成功生成: {processed_count}")
    print(f"跳过(无效): {skipped_count}")
    print(f"文件保存于: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()