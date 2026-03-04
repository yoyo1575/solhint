import json
import os
import random
from typing import Optional, Dict, Any
from solc_windows import SolidityLinter
import time

def lintseq_backward_sampling_to_empty(code: str, seed: int = 42, max_iterations: int = 2000) -> Optional[Dict[str, Any]]:
    """
    执行 backward sampling：
    1. 随机删除一行代码（优先保留 pragma solidity 声明）。
    2. 进入修复循环：如果存在错误，删除所有报错行，直到代码再次合法（或为空）。
    3. 记录合法快照。
    4. 循环直到代码完全为空。
    """
    linter = SolidityLinter()
    random.seed(seed)

    current_code = code.strip()

    # 1. 检查原始代码是否可编译
    try:
        errors = linter.get_error_lines_only_error_3(current_code)
        # if errors:
        #     print("Initial code has errors, attempting to repair first...")
        #     # 如果初始代码有错，交由下方的 while 循环处理
    except Exception as e:
        print(f"Linter error on initial code: {e}")
        return None

    # 记录所有合法快照
    valid_snapshots = [current_code]

    iteration_count = 0

    while current_code.strip() and iteration_count < max_iterations:
        iteration_count += 1
        lines_list = current_code.splitlines()

        # 如果全空，跳出
        if not any(line.strip() for line in lines_list):
            break

        # ==========================================
        # 步骤 1: 随机删除一行 (Perturbation)
        # 逻辑更新：Pragma 最后删除
        # ==========================================

        # 1.1 找到所有非空行的索引
        non_empty_indices = [
            i for i, line in enumerate(lines_list)
            if line.strip()
        ]

        if not non_empty_indices:
            break

        # 1.2 寻找 pragma solidity 行的位置
        pragma_idx = -1
        for i in non_empty_indices:
            if lines_list[i].strip().startswith("pragma solidity"):
                pragma_idx = i
                break

        # 1.3 构建候选删除列表
        candidate_indices = []

        if pragma_idx != -1:
            # 如果存在 pragma 行
            others = [x for x in non_empty_indices if x != pragma_idx]
            if others:
                # 如果还有其他代码行，只从其他行中选择，保护 pragma
                candidate_indices = others
            else:
                # 如果只剩 pragma 行了（others为空），那么只能删 pragma
                candidate_indices = [pragma_idx]
        else:
            # 没有 pragma，所有非空行都可以删
            candidate_indices = non_empty_indices

        # 随机选择一行删除
        idx_to_remove = random.choice(candidate_indices)
        lines_list.pop(idx_to_remove)

        # 更新当前代码
        current_code = "\n".join(lines_list)

        # ==========================================
        # 步骤 2: 修复循环 (Repair Loop)
        # ==========================================
        repair_steps = 0
        max_repair_steps = 50
        is_valid_state = False

        while repair_steps < max_repair_steps:
            # 如果代码删空了，视为合法
            if not current_code.strip():
                is_valid_state = True
                break

            try:
                # 获取报错行号
                errors = linter.get_error_lines_only_error_3(current_code)

                if not errors:
                    # 无错 -> 修复成功
                    is_valid_state = True
                    break

                # 有错 -> 删除报错行
                unique_errors = sorted(list(set(errors)), reverse=True)

                current_lines = current_code.splitlines()
                num_lines_now = len(current_lines)

                lines_removed_in_repair = False
                for line_num in unique_errors:
                    idx = line_num - 1  # 1-based to 0-based

                    if 0 <= idx < num_lines_now:
                        del current_lines[idx]
                        lines_removed_in_repair = True

                # 防死循环兜底：如果 Linter 报错但行号无效，强制删最后一行
                if not lines_removed_in_repair:
                    if current_lines:
                        current_lines.pop()

                current_code = "\n".join(current_lines)
                repair_steps += 1

            except Exception as e:
                print(f"Error during repair loop: {e}")
                current_code = ""
                break

        # ==========================================
        # 步骤 3: 记录快照
        # ==========================================
        if is_valid_state:
            # 只有当代码发生变化时才记录
            if not valid_snapshots or valid_snapshots[-1] != current_code:
                valid_snapshots.append(current_code)

        # 检查是否完全空了
        if not current_code.strip():
            # 确保空字符串被记录
            if valid_snapshots and valid_snapshots[-1] != "":
                valid_snapshots.append("")
            break

    return {
        "initial_code": code,
        "final_code": current_code,
        "valid_snapshots": valid_snapshots,
        "total_steps": len(valid_snapshots) - 1
    }


def test_specific_line(jsonl_file: str, line_no: int, output_dir: str, seed: int = 42):
    target_item = None
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if idx == line_no:
                try:
                    target_item = json.loads(line.strip())
                except json.JSONDecodeError:
                    return None
                break
        else:
            return None

    code = target_item.get("contract", "").strip()
    if not code:
        return None

    print(f"Processing Line {line_no}, Code Length: {len(code)}")
    # 调用新的逻辑
    result = lintseq_backward_sampling_to_empty(code, seed=seed)

    if result:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"contract_{line_no}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result
    else:
        return None


def random_line_from_jsonl(jsonl_file):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)

    if total_lines == 0:
        print("empty line")
        return None

    random_line_num = random.randint(1, total_lines)
    return random_line_num

if __name__ == "__main__":
    # 配置区域
    jsonl_file = r"D:\paper\Synthtic_edit_sequence\lintseq\solhint\raw_sol\merge_clean_sol.jsonl"
    output_dir = r"D:\paper\Synthtic_edit_sequence\lintseq\solhint\gen_sol_specific"

    line_no = 4

    current_seed = int(time.time())
    result = test_specific_line(
        jsonl_file=jsonl_file,
        line_no=line_no,
        output_dir=output_dir,
        seed=current_seed # 时间戳 完全随机删除路径
    )
