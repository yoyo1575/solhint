import subprocess
import tempfile
import os
import re
from packaging import version
from typing import Dict, Optional, Tuple, List


class SolidityLinter:
    """
    Solidity 语法检查器：严格根据 pragma 语义选择 solc 版本，
    如果没有找到匹配版本 → 直接报错，不 fallback。
    """

    def __init__(self, solc_versions_dir: str = r"D:\solc"):
        self.solc_versions_dir = solc_versions_dir
        self.available_versions: Dict[str, str] = {}  # { "0.6.12": "path/to/solc.exe" }

        self._discover_versions()

        if not self.available_versions:
            raise RuntimeError(f" 在目录中找不到 solc: {solc_versions_dir}")

        # 预排序可用版本
        self.sorted_versions = sorted(
            [version.parse(v) for v in self.available_versions.keys()]
        )

    # 搜索 solc 可执行文件
    def _discover_versions(self):
        if not os.path.exists(self.solc_versions_dir):
            raise RuntimeError(f"solc 目录不存在: {self.solc_versions_dir}")

        for item in os.listdir(self.solc_versions_dir):
            full_path = os.path.join(self.solc_versions_dir, item)

            # 1. 单文件形式：solc-v0.6.12.exe
            if os.path.isfile(full_path) and item.endswith(".exe"):
                self._process_exe_file(item, full_path)

            # 2. 文件夹形式：solc-windows-0.6.12/
            elif os.path.isdir(full_path):
                version_match = re.search(r"(\d+\.\d+\.\d+)", item)
                if version_match:
                    version_str = version_match.group(1)
                    self._process_version_folder(version_str, full_path)

    def _process_exe_file(self, filename: str, full_path: str):
        m = re.search(r"(\d+\.\d+\.\d+)", filename)
        if m:
            v = m.group(1)
            self.available_versions[v] = full_path

    def _process_version_folder(self, version: str, folder_path: str):
        for file in os.listdir(folder_path):
            if file.lower() in ["solc.exe", "solc-windows.exe"]:
                self.available_versions[version] = os.path.join(folder_path, file)
                return

    # 从 pragma solidity 解析版本范围
    PRAGMA = re.compile(r"pragma\s+solidity\s+([^;]+);", re.IGNORECASE)

    def _extract_pragma(self, code: str) -> Optional[str]:
        m = self.PRAGMA.search(code)
        return m.group(1).strip() if m else None

    def _parse_range(self, pragma_expr: str) -> Tuple[List[Tuple[str, version.Version]], List[Tuple[str, version.Version]]]:
        """
        将表达式转化为上下界条件，例如：
            >=0.6.0  <0.7.0
            ^0.5.2   -> >=0.5.2 <0.6.0
            0.6.12   -> =0.6.12
        """

        tokens = pragma_expr.split()
        lowers = []
        uppers = []

        for tok in tokens:

            # caret: ^0.6.2  => >=0.6.2 <0.7.0
            if tok.startswith("^"):
                base = version.parse(tok[1:])
                major, minor, patch = base.major, base.minor, base.micro
                lowers.append((">=", base))
                uppers.append(("<", version.Version(f"{major}.{minor + 1}.0")))
                continue

            # 无操作符：0.6.12  等价于 =0.6.12
            if re.fullmatch(r"\d+\.\d+\.\d+", tok):
                lowers.append(("=", version.parse(tok)))
                continue

            # >=, >, <=, <, = 形式
            m = re.match(r"(>=|<=|>|<|=)\s*(\d+\.\d+\.\d+)", tok)
            if m:
                op = m.group(1)
                v = version.parse(m.group(2))
                if op in [">=", ">"]:
                    lowers.append((op, v))
                else:
                    uppers.append((op, v))

        return lowers, uppers


    def _satisfy(self, v: version.Version,lowers: List[Tuple[str, version.Version]],uppers: List[Tuple[str, version.Version]]) -> bool:

        for op, bound in lowers:
            if op == ">=" and not (v >= bound):
                return False
            if op == ">" and not (v > bound):
                return False
            if op == "=" and not (v == bound):
                return False

        for op, bound in uppers:
            if op == "<=" and not (v <= bound):
                return False
            if op == "<" and not (v < bound):
                return False

        return True
    # 找到严格符合 pragma 的 solc 版本

    def _find_exact_solc(self, pragma_expr: Optional[str]) -> str:
        if not pragma_expr:
            raise RuntimeError("源码未声明 pragma solidity，无法选择 solc 版本")

        lowers, uppers = self._parse_range(pragma_expr)

        candidates = []
        for v in self.sorted_versions:
            if self._satisfy(v, lowers, uppers):
                candidates.append(v)

        if not candidates:
            raise RuntimeError(f" 找不到符合 pragma `{pragma_expr}` 的 solc 版本")

        # 选最大的（最新的）合法版本
        chosen = max(candidates)
        return self.available_versions[str(chosen)]

    # -------------------------------------------------------
    # 核心语法检查
    # -------------------------------------------------------
    def get_error_lines(self, code: str) -> list:
        pragma_expr = self._extract_pragma(code)
        print(f"pragma 要求: {pragma_expr}")

        solc_path = self._find_exact_solc(pragma_expr)
        print(f"使用 solc: {solc_path}")

        # 写临时文件
        with tempfile.NamedTemporaryFile("w", suffix=".sol", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmpfile = f.name

        try:
            result = subprocess.run(
                [solc_path, tmpfile],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            stderr = result.stderr
            print(stderr)

            lines = set(re.findall(r":(\d+):\d+:", stderr))
            return sorted({int(x) for x in lines})

        finally:
            os.unlink(tmpfile)

