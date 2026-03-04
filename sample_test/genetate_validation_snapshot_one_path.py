import json
import os
import random
import multiprocessing
from tqdm import tqdm
from typing import Dict, Any, Optional

# 确保 solc_windows.py 在同一目录下
try:
    from solc_windows import SolidityLinter
except ImportError:
    print("错误: 找不到 solc_windows.py，请确保它在当前目录下！")
    exit(1)


# ======================================================
# 核心逻辑: lintseq_backward_sampling_to_empty
# ======================================================
def lintseq_backward_sampling_to_empty(code: str, seed: int = 42, max_iterations: int = 2000) -> Optional[Dict[str, Any]]:
    # 在多进程中，每个进程独立实例化 Linter
    try:
        linter = SolidityLinter()
    except Exception as e:
        # 如果 Linter 初始化失败，返回 None
        return None

    random.seed(seed)
    current_code = code.strip()

    # 1. 检查原始代码是否可编译
    try:
        # 使用 _3 版本 (即你最新的 Version 4 逻辑)
        errors = linter.get_error_lines_only_error_3(current_code)
        # 即使初始有错，我们也继续，让 repair loop 去修
    except Exception:
        return None

    valid_snapshots = [current_code]
    iteration_count = 0

    while current_code.strip() and iteration_count < max_iterations:
        iteration_count += 1
        lines_list = current_code.splitlines()

        if not any(line.strip() for line in lines_list):
            break

        # --- 步骤 1: 随机删除 ---
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

        # --- 步骤 2: 修复循环 ---
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

        # --- 步骤 3: 记录快照 ---
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


# ======================================================
# Worker 函数: 只负责计算，不负责写文件
# ======================================================
def process_single_item(args):
    idx, line_str, seed_base = args

    try:
        item = json.loads(line_str)
        code = item.get("contract", "").strip()

        if not code:
            return {"status": "empty_code"}

        current_seed = seed_base + idx
        # 执行采样
        result = lintseq_backward_sampling_to_empty(code, seed=current_seed)

        if result:
            # 返回计算结果给主进程
            return {"status": "success", "data": result, "line_no": idx}
        else:
            return {"status": "failed_logic"}

    except Exception as e:
        return {"status": "error", "msg": str(e)}


def count_lines(filename):
    with open(filename, "rb") as f:
        return sum(1 for _ in f)


# ======================================================
# 主函数
# ======================================================
def main():
    # --- 配置 ---
    # 输入文件路径
    input_file = r"D:\paper\Synthtic_edit_sequence\lintseq\solhint\raw_sol\merge_clean_sol.jsonl"

    # 输出文件路径 (合并为一个大的 jsonl)
    output_file = r"D:\paper\Synthtic_edit_sequence\lintseq\solhint\gen_sol_validation_snapshot"

    seed_base = 42
    num_processes = max(1, multiprocessing.cpu_count() - 2)
    # -----------

    if not os.path.exists(input_file):
        print(f"Error: 输入文件不存在: {input_file}")
        return

    # 简单的断点续传检查：计算已处理的行数
    processed_count = 0
    if os.path.exists(output_file):
        print(f"检测到输出文件 {output_file} 已存在，正在计算已处理行数...")
        processed_count = count_lines(output_file)
        print(f"已处理: {processed_count} 行，将跳过前 {processed_count} 个任务。")

    print(f"Counting lines in {input_file}...")
    total_lines = count_lines(input_file)
    print(f"Total contracts: {total_lines}. Processes: {num_processes}")

    # 读取任务
    tasks = []
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            # 如果该行已经处理过，就不加入任务列表
            if idx > processed_count:
                tasks.append((idx, line, seed_base))

    print(f"Remaining tasks to process: {len(tasks)}")

    stats = {
        "success": 0, "failed_logic": 0,
        "empty_code": 0, "errors": 0
    }

    # 打开输出文件 (追加模式 'a')
    with open(output_file, "a", encoding="utf-8") as f_out:

        with multiprocessing.Pool(processes=num_processes) as pool:
            # imap_unordered 让结果哪个先算完就返回哪个
            for res in tqdm(pool.imap_unordered(process_single_item, tasks, chunksize=1), total=len(tasks)):

                status = res["status"]

                if status == "success":
                    stats["success"] += 1
                    # 写入一行 JSON 到大文件
                    data = res["data"]
                    # 也可以加上原来的行号方便追溯
                    data["original_line_no"] = res["line_no"]
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    # 立即刷新缓冲区，防止断电丢失太多
                    f_out.flush()

                elif status == "failed_logic":
                    stats["failed_logic"] += 1
                elif status == "empty_code":
                    stats["empty_code"] += 1
                elif status == "error":
                    stats["errors"] += 1

    print("\nProcessing complete!")
    print("Statistics (Current Run):")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Data saved to: {output_file}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()