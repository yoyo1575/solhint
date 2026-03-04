
# -*- coding: utf-8 -*-

import json
import hashlib
from pathlib import Path
from collections import Counter

# ======== 路径 ========
BASE_DIR = Path(r"D:\paper\Synthtic_edit_sequence\lintseq\solhint\DATA\hugging_data\jsonl\test")
INPUT_FILES = [
    BASE_DIR / "test_0.jsonl",
    BASE_DIR / "test_1.jsonl",
    BASE_DIR / "test_2.jsonl",
]
OUT_FILE = Path(r"D:\paper\Synthtic_edit_sequence\lintseq\solhint\DATA\merge_test_data\test_merged.jsonl")

# ======== 去重键（优先 address+func_name，其次 file_path+func_name，最后 code hash）========
def stable_key(ex: dict) -> str:
    addr = (ex.get("contract_address") or "").strip().lower()
    fn = (ex.get("func_name") or "").strip()
    fp = (ex.get("file_path") or "").strip()

    if addr and fn:
        return f"addr:{addr}::fn:{fn}"
    if fp and fn:
        return f"file:{fp}::fn:{fn}"

    code = ex.get("class_code") or ex.get("func_code") or ""
    code = code.replace("\r\n", "\n").strip()
    h = hashlib.sha256(code.encode("utf-8")).hexdigest()
    return f"codehash:{h}"

def main():
    stats = Counter()
    seen = {}  # key -> record
    bad_lines = []

    # 校验输入文件存在
    for fp in INPUT_FILES:
        if not fp.exists():
            raise FileNotFoundError(f"Input file not found: {fp}")

    for fp in INPUT_FILES:
        stats["files"] += 1
        with fp.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                stats["lines"] += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except Exception:
                    stats["bad_json"] += 1
                    bad_lines.append(f"{fp}:{ln}")
                    continue

                stats["records_in"] += 1
                k = stable_key(ex)
                if k in seen:
                    stats["dups"] += 1
                    # 保留第一次出现（如果你想保留最后一次，把下面两行取消注释）
                    # seen[k] = ex
                else:
                    seen[k] = ex

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as w:
        for ex in seen.values():
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")

    stats["records_out"] = len(seen)

    print("=== Merge Done ===")
    print("Input files:", [str(p) for p in INPUT_FILES])
    print("Output file:", str(OUT_FILE))
    print("Lines read:", stats["lines"])
    print("Records in:", stats["records_in"])
    print("Duplicates removed:", stats["dups"])
    print("Bad JSON lines:", stats["bad_json"])
    print("Records out:", stats["records_out"])

    if bad_lines:
        print("\nBad JSON examples (first 20):")
        for x in bad_lines[:20]:
            print("  ", x)

if __name__ == "__main__":
    main()