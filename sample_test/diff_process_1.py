import json
import os
import difflib
from tqdm import tqdm

# ================= 论文主实验配置（改成你的 test 路径） =================
# 输入：3-path
INPUT_FILE = r"D:\paper\Synthtic_edit_sequence\lintseq\solhint\DATA\3_3path_process_merge_test_data\3path_process_merge_test_data.jsonl"

# 输出：test 的 diff
OUTPUT_FILE = r"D:\paper\Synthtic_edit_sequence\lintseq\solhint\DATA\4_process_merge_3path_diff_data\3path_process_merge_test_data_diff.jsonl"

# 训练集风格：直接用换行拼接多个 step diff
JOIN_SEPARATOR = "\n"
# ======================================================================


def generate_unified_diff_hunks_only(str_a: str, str_b: str) -> str:
    """
    生成与你训练集一致的 diff：只保留 @@ hunks 和 +/-/上下文行，不包含 ---/+++ 文件头。
    """
    a_lines = str_a.splitlines(keepends=True)
    b_lines = str_b.splitlines(keepends=True)

    diff = difflib.unified_diff(
        a_lines, b_lines,
        fromfile="prev", tofile="curr",
        n=3
    )

    diff_content = []
    for line in diff:
        # 过滤掉文件头（训练集就是这种风格）
        if line.startswith('---') or line.startswith('+++'):
            continue
        diff_content.append(line)

    return "".join(diff_content).strip()


def ensure_trailing_newline(s: str) -> str:
    """保证末尾有换行，避免 diff 末尾粘连/格式不稳定"""
    if s is None:
        return ""
    if s != "" and not s.endswith("\n"):
        return s + "\n"
    return s


def main():
    # 确保输出目录存在
    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print("🚀 Processing diffs (paper-main, hunks-only)...")
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for line in tqdm(lines, desc="Processing"):
            try:
                item = json.loads(line)
                snapshots = item.get("valid_snapshots", [])

                # 无快照或无变化 → 跳过
                if not snapshots or len(snapshots) < 2:
                    skipped_count += 1
                    continue

                # 1) 反转：空 -> 完整
                forward_snapshots = snapshots[::-1]

                edit_sequence_parts = []

                # 2) 逐步生成 hunks-only diff
                for i in range(len(forward_snapshots) - 1):
                    prev_state = ensure_trailing_newline(forward_snapshots[i])
                    curr_state = ensure_trailing_newline(forward_snapshots[i + 1])

                    diff_text = generate_unified_diff_hunks_only(prev_state, curr_state)
                    if diff_text:
                        edit_sequence_parts.append(diff_text)

                if not edit_sequence_parts:
                    skipped_count += 1
                    continue

                full_edit_sequence = JOIN_SEPARATOR.join(edit_sequence_parts)

                # 3) 补全 final_code（完整代码）
                # 你 snapshots 里 final_code 往往是空串，但 initial_code / snapshots[0] 是完整代码
                final_code = item.get("final_code") or item.get("initial_code") or snapshots[0]

                # 4) 写出（保留 path_id，方便论文分析与追溯）
                output_item = {
                    "original_line_no": item.get("original_line_no"),
                    "path_id": item.get("path_id"),
                    "final_code": final_code,
                    "edit_sequence": full_edit_sequence,
                    "steps_count": len(edit_sequence_parts)
                }

                f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                processed_count += 1

            except Exception:
                error_count += 1
                continue

    print("✅ Done!")
    print(f"Total lines : {len(lines)}")
    print(f"Processed   : {processed_count}")
    print(f"Skipped     : {skipped_count}")
    print(f"Errors      : {error_count}")
    print(f"Saved to    : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()