import json
import os
import random
from typing import Optional, Dict, Any, List
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# 隔一级用一个".." 两个用两个 "..",".."
target_dir = os.path.abspath(os.path.join(current_dir, ".."))
if target_dir not in sys.path:
    sys.path.append(target_dir)




# 依赖检查
try:
    from solc_windows import SolidityLinter
except ImportError:
    print("错误: 找不到 solc_windows.py，请确保它在当前目录下！")
    exit(1)

# ==========================================
# 配置参数
# ==========================================
NUM_PATHS = 3  # 生成几条路径
SEED_BASE = 42  # 种子基数
MIN_CODE_LINES = 6  # 最小行数限制


# ==========================================
# 核心逻辑 (保持与批量脚本一致)
# ==========================================
def lintseq_backward_sampling_to_empty(code: str, seed: int = 42, max_iterations: int = 2000) -> Optional[Dict[str, Any]]:
    try:
        linter = SolidityLinter()
    except Exception as e:
        print(f"Linter Init Error: {e}")
        return None

    random.seed(seed)
    current_code = code.strip()

    # 初始检查
    try:
        errors = linter.get_error_lines_only_error_3(current_code)
    except Exception:
        return None

    valid_snapshots = [current_code]
    iteration_count = 0

    while current_code.strip() and iteration_count < max_iterations:
        iteration_count += 1
        lines_list = current_code.splitlines()

        if not any(line.strip() for line in lines_list):
            break

        # 1. 随机删除
        non_empty_indices = [i for i, line in enumerate(lines_list) if line.strip()]
        if not non_empty_indices:
            break

        # 保护 pragma
        pragma_idx = -1
        for i in non_empty_indices:
            if lines_list[i].strip().startswith("pragma solidity"):
                pragma_idx = i
                break

        candidate_indices = []
        if pragma_idx != -1:
            others = [x for x in non_empty_indices if x != pragma_idx]
            if others:
                candidate_indices = others
            else:
                candidate_indices = [pragma_idx]
        else:
            candidate_indices = non_empty_indices

        idx_to_remove = random.choice(candidate_indices)
        lines_list.pop(idx_to_remove)
        current_code = "\n".join(lines_list)

        # 2. 修复循环
        repair_steps = 0
        max_repair_steps = 50
        is_valid_state = False

        while repair_steps < max_repair_steps:
            if not current_code.strip():
                is_valid_state = True
                break

            try:
                errors = linter.get_error_lines_only_error_3(current_code)

                if not errors:
                    is_valid_state = True
                    break

                unique_errors = sorted(list(set(errors)), reverse=True)
                current_lines = current_code.splitlines()
                num_lines_now = len(current_lines)

                lines_removed_in_repair = False
                for line_num in unique_errors:
                    idx = line_num - 1
                    if 0 <= idx < num_lines_now:
                        del current_lines[idx]
                        lines_removed_in_repair = True

                if not lines_removed_in_repair:
                    if current_lines:
                        current_lines.pop()

                current_code = "\n".join(current_lines)
                repair_steps += 1

            except Exception:
                current_code = ""
                break

        # 3. 记录
        if is_valid_state:
            if not valid_snapshots or valid_snapshots[-1] != current_code:
                valid_snapshots.append(current_code)

        if not current_code.strip():
            if valid_snapshots and valid_snapshots[-1] != "":
                valid_snapshots.append("")
            break

    return {
        "initial_code": code,
        "final_code": current_code,
        "valid_snapshots": valid_snapshots,
        "total_steps": len(valid_snapshots) - 1
    }


# ==========================================
# 单样本多路径测试函数
# ==========================================
def test_specific_line_multipath(jsonl_file: str, line_no: int, output_dir: str):
    # 1. 读取指定行
    target_item = None
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if idx == line_no:
                try:
                    target_item = json.loads(line.strip())
                except json.JSONDecodeError:
                    print("Error: JSON Decode Error")
                    return None
                break
        else:
            print(f"Error: Line {line_no} not found.")
            return None

    code = target_item.get("contract", "").strip()
    if not code:
        print("Error: Empty code.")
        return None

    # 2. 检查长度
    print(f"原始代码行数: {len(code.splitlines())}")
    if len(code.splitlines()) < MIN_CODE_LINES:
        print(f"警告: 代码行数少于 {MIN_CODE_LINES}，在批量脚本中会被跳过。但此处强制运行测试。")

    print(f"正在为第 {line_no} 行样本生成 {NUM_PATHS} 条不同路径...\n")

    generated_paths = []

    # 3. 循环生成多条路径
    for path_i in range(NUM_PATHS):
        # 计算种子 (与批量脚本逻辑一致)
        current_seed = SEED_BASE + (line_no * 1000) + path_i
        print(f"Path {path_i + 1}: 使用种子 {current_seed} ...")

        result = lintseq_backward_sampling_to_empty(code, seed=current_seed)

        if result:
            print(f"steps: {result['total_steps']}")
            # 标记 ID
            result["path_id"] = path_i
            generated_paths.append(result)
        else:
            print(f"失败")

    # 4. 保存结果
    if generated_paths:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"contract_{line_no}_3paths.json"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(generated_paths, f, indent=2, ensure_ascii=False)

        print(f"\nsave to : {output_path}")

        # 5. 简单的多样性检查
        print("\n--- 多样性检查 ---")
        steps_list = [p['total_steps'] for p in generated_paths]
        print(f"三条路径的步数分别为: {steps_list}")

        if len(set(steps_list)) > 1:
            print("验证通过：路径步数不同，说明生成了不同的序列。")
        else:
            # 如果步数相同，检查具体的快照内容是否完全一致
            path0_snaps = generated_paths[0]['valid_snapshots']
            path1_snaps = generated_paths[1]['valid_snapshots']
            if path0_snaps == path1_snaps:
                print("警告：生成的路径完全相同 可能是代码太短，或者随机性不足。")
            else:
                print("验证通过：虽然步数相同，但中间快照内容不同")


if __name__ == "__main__":
    jsonl_file = r"D:\paper\Synthtic_edit_sequence\lintseq\solhint\raw_sol\hugging_data_0.jsonl"
    output_dir = r"D:\paper\Synthtic_edit_sequence\lintseq\solhint\gen_sol_validation_snapshot_multi"

    line_no = 333

    test_specific_line_multipath(jsonl_file, line_no, output_dir)