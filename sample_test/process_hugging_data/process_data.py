import json
import re
from tqdm import tqdm


def extract_version(version_str):
    """提取版本号"""
    if not version_str:
        return None
    match = re.search(r"(\d+\.\d+\.\d+)", version_str)
    if match:
        return match.group(1)
    return None


def remove_comments(string):
    """
    强力去除 Solidity 注释 (支持中文、特殊字符、跨行)
    保持字符串内容的完整性
    """
    # 匹配规则：
    # Group 1: 双引号或单引号字符串 (被保护，不删)
    # Group 2: 单行注释 //... 或 多行注释 /*...*/ (要删)
    pattern = r'("(?:[^"\\]|\\.)*"|\'(?:[^"\\]|\\.)*\')|((?://[^\r\n]*)|(?:\/\*[\s\S]*?\*\/))'
    regex = re.compile(pattern)

    def _replacer(match):
        if match.group(2):
            return ""  # 如果是注释，替换为空
        else:
            return match.group(1)  # 如果是字符串，保留

    return regex.sub(_replacer, string)


def preprocess_dataset_simple(input_file, output_file):
    #print(f"开始极简清洗: {input_file} -> {output_file}")

    unique_contracts = set()
    valid_count = 0
    skipped_dup = 0

    with open(input_file, "r", encoding="utf-8", errors='replace') as f_in, \
            open(output_file, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in):
            try:
                line = line.strip()
                if not line: continue

                item = json.loads(line)

                # 兼容不同数据源字段
                raw_code = item.get("class_code") or item.get("contract") or ""
                if not raw_code:
                    continue

                # 1. 去注释
                clean_code = remove_comments(raw_code)

                # 2. 清理空行 (这一步让代码非常紧凑，适合后续逐行删除)
                lines = [line.rstrip() for line in clean_code.splitlines()]
                non_empty_lines = [line for line in lines if line.strip()]
                final_code_body = "\n".join(non_empty_lines)

                if not final_code_body:
                    continue

                # 3. 补 Pragma
                has_pragma = "pragma solidity" in final_code_body.lower()
                final_output_code = final_code_body

                if not has_pragma:
                    # 尝试从各个可能的字段找版本号
                    raw_ver = item.get("compiler_version") or \
                              item.get("original_meta", {}).get("version") or \
                              item.get("version") or ""

                    clean_ver = extract_version(raw_ver)

                    if clean_ver:
                        final_output_code = f"pragma solidity ^{clean_ver};\n" + final_code_body
                    else:
                        # 实在找不到，给个通用的 0.4.0 兜底
                        final_output_code = "pragma solidity ^0.4.0;\n" + final_code_body

                # 4. 去重
                code_hash = hash(final_output_code)
                if code_hash in unique_contracts:
                    skipped_dup += 1
                    continue
                unique_contracts.add(code_hash)

                # 5. 极简保存：只存 contract 字段
                new_item = {
                    "contract": final_output_code
                }

                f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                valid_count += 1

            except Exception:
                continue

    print("\n清洗完成！")
    print(f"  有效数据量: {valid_count}")
    print(f"  去除重复量: {skipped_dup}")


if __name__ == "__main__":
    # 请修改路径
    input_jsonl = r"D:\paper\Synthtic_edit_sequence\lintseq\solhint\DATA\hugging_data\jsonl\train\train-00001-of-00021.jsonl"

    # 这个文件将只包含 {"contract": "..."}，非常纯净
    output_jsonl = r"D:\paper\Synthtic_edit_sequence\lintseq\solhint\raw_sol\hugging_data_1.jsonl"

    preprocess_dataset_simple(input_jsonl, output_jsonl)