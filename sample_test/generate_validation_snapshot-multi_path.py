import json
import os
import random
import multiprocessing
from tqdm import tqdm
from typing import Dict, Any, Optional, List

# ======================================================
# 依赖检查
# ======================================================
try:
    from solc_windows import SolidityLinter
except ImportError:
    print("错误: 找不到 solc_windows.py，请确保它在当前目录下！")
    exit(1)

# ======================================================
# 全局配置参数
# ======================================================
NUM_PATHS_PER_SAMPLE = 3  # 每个样本生成几条不同的路径
MIN_CODE_LINES = 6  # 最小行数限制：少于此行数的代码将跳过
MAX_ITERATIONS = 2000  # 最大迭代次数，防止死循环
SEED_BASE = 42  # 随机种子基数


# ======================================================
# 核心逻辑: Backward Sampling
# ======================================================
def lintseq_backward_sampling_to_empty(code: str, seed: int = 42, max_iterations: int = 2000) -> Optional[
    Dict[str, Any]]:
    """
    执行 backward sampling：
    1. 随机删除一行代码（优先保留 pragma solidity 声明）。
    2. 进入修复循环：如果存在错误，删除所有报错行，直到代码再次合法（或为空）。
    3. 记录合法快照。
    4. 循环直到代码完全为空。
    """
    # 在多进程中，每个进程独立实例化 Linter
    try:
        linter = SolidityLinter()
    except Exception:
        # 如果 Linter 初始化失败（如找不到 solc），直接返回 None
        return None

    random.seed(seed)
    current_code = code.strip()

    # 1. 检查原始代码是否可编译
    try:
        # 使用 _3 版本 (即最新的健壮版本)
        errors = linter.get_error_lines_only_error_3(current_code)
        # 这里不做 print，保持静默。即使有错，也会进入下方的 repair loop 修复。
    except Exception:
        return None

    valid_snapshots = [current_code]
    iteration_count = 0

    while current_code.strip() and iteration_count < max_iterations:
        iteration_count += 1
        lines_list = current_code.splitlines()

        # 如果全空，跳出
        if not any(line.strip() for line in lines_list):
            break

        # --- 步骤 1: 随机删除一行 (Perturbation) ---
        non_empty_indices = [i for i, line in enumerate(lines_list) if line.strip()]
        if not non_empty_indices:
            break

        # 保护 pragma 行，尽量把它留到最后删
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
                candidate_indices = [pragma_idx]  # 只剩 pragma 时才删它
        else:
            candidate_indices = non_empty_indices

        idx_to_remove = random.choice(candidate_indices)
        lines_list.pop(idx_to_remove)
        current_code = "\n".join(lines_list)

        # --- 步骤 2: 修复循环 (Repair Loop) ---
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
                    idx = line_num - 1  # 1-based to 0-based
                    if 0 <= idx < num_lines_now:
                        del current_lines[idx]
                        lines_removed_in_repair = True

                # 兜底：如果报错但没删掉任何行（可能是行号解析失败），强制删最后一行防止死循环
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
            # 只有当代码发生实质变化时才记录
            if not valid_snapshots or valid_snapshots[-1] != current_code:
                valid_snapshots.append(current_code)

        # 检查是否完全空了
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
# Worker 函数: 多路径生成逻辑
# ======================================================
def process_single_item_multipath(args):
    idx, line_str, seed_base = args

    try:
        item = json.loads(line_str)
        code = item.get("contract", "").strip()

        # 1. 判空
        if not code:
            return {"status": "empty_code"}

        # 2. 行数过滤: 太短的代码直接跳过
        if len(code.splitlines()) < MIN_CODE_LINES:
            return {"status": "skipped_too_short"}

        generated_paths = []

        # 3. 循环生成 N 条路径
        for path_i in range(NUM_PATHS_PER_SAMPLE):
            # 种子设计：保证同一文件的不同路径种子不同，且全局唯一
            current_seed = seed_base + (idx * 1000) + path_i

            result = lintseq_backward_sampling_to_empty(
                code,
                seed=current_seed,
                max_iterations=MAX_ITERATIONS
            )

            if result:
                # 标记元数据
                result["original_line_no"] = idx
                result["path_id"] = path_i
                generated_paths.append(result)

        if len(generated_paths) > 0:
            return {"status": "success", "data_list": generated_paths}
        else:
            return {"status": "failed_logic"}

    except Exception as e:
        return {"status": "error", "msg": str(e)}


def count_lines(filename):
    """快速计算文件行数"""
    if not os.path.exists(filename):
        return 0
    with open(filename, "rb") as f:
        return sum(1 for _ in f)


# ======================================================
# 主函数
# ======================================================
def main():
    # --- 路径配置 ---
    # 请根据实际情况修改
    input_file = r"D:\yoyo\sythtic_edit_sequence\lintseq\solhint\data\merge_clean_sol.jsonl"

    # 输出文件 (必须带后缀 .jsonl)
    output_file = r"D:\yoyo\sythtic_edit_sequence\lintseq\solhint\gen_sol_dataset_3paths.jsonl"

    # 进程数：CPU核心数 - 2
    num_processes = max(1, multiprocessing.cpu_count() - 2)
    # ----------------

    if not os.path.exists(input_file):
        print(f"Error: 输入文件不存在: {input_file}")
        return

    # 断点续传逻辑
    processed_count = 0
    if os.path.exists(output_file):
        print(f"检测到输出文件 {output_file} 已存在。")
        current_lines = count_lines(output_file)
        # 估算已处理的原始文件数 (假设每个成功的文件都写了 NUM_PATHS_PER_SAMPLE 行)
        # 注意：这只是估算，如果之前有failed的，可能会少跳过一点，这是安全的
        processed_count = current_lines // NUM_PATHS_PER_SAMPLE
        print(f"当前输出文件有 {current_lines} 行。")
        print(f"将尝试跳过前 {processed_count} 个原始合约任务。")

    print(f"Counting lines in {input_file}...")
    total_lines = count_lines(input_file)
    print(f"Total contracts: {total_lines}. Target Paths per contract: {NUM_PATHS_PER_SAMPLE}")

    # 准备任务
    tasks = []
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if idx > processed_count:
                tasks.append((idx, line, SEED_BASE))

    print(f"Remaining tasks to process: {len(tasks)}")
    print(f"Starting processing with {num_processes} processes...")

    stats = {
        "success_files": 0,
        "total_paths_generated": 0,
        "skipped_too_short": 0,
        "failed_files": 0,
        "empty_code": 0,
        "errors": 0
    }

    # 打开输出文件 (追加模式 'a')
    with open(output_file, "a", encoding="utf-8") as f_out:
        with multiprocessing.Pool(processes=num_processes) as pool:
            # imap_unordered 实时返回结果
            for res in tqdm(pool.imap_unordered(process_single_item_multipath, tasks, chunksize=1), total=len(tasks)):

                status = res["status"]

                if status == "success":
                    stats["success_files"] += 1
                    data_list = res["data_list"]
                    stats["total_paths_generated"] += len(data_list)

                    # 写入生成的所有路径
                    for data in data_list:
                        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    f_out.flush()

                elif status == "skipped_too_short":
                    stats["skipped_too_short"] += 1
                elif status == "failed_logic":
                    stats["failed_files"] += 1
                elif status == "empty_code":
                    stats["empty_code"] += 1
                elif status == "error":
                    stats["errors"] += 1

    print("\nProcessing complete!")
    print("Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Data saved to: {output_file}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()