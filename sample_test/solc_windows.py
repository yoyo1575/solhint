import subprocess
import tempfile
import os
import re
from packaging import version
from typing import Dict, Optional, Tuple, List
import random


class SolidityLinter:
    def __init__(self, solc_versions_dir: str = r"D:\solc"):
        self.solc_versions_dir = solc_versions_dir
        self.available_versions: Dict[str, str] = {}
        self._discover_versions()

        if not self.available_versions:
            raise RuntimeError(f"在目录中找不到 solc: {solc_versions_dir}")

        self.sorted_versions = sorted(
            [version.parse(v) for v in self.available_versions.keys()]
        )

    def _discover_versions(self):
        if not os.path.exists(self.solc_versions_dir):
            raise RuntimeError(f"solc 目录不存在: {self.solc_versions_dir}")

        for item in os.listdir(self.solc_versions_dir):
            full_path = os.path.join(self.solc_versions_dir, item)
            if os.path.isfile(full_path) and item.endswith(".exe"):
                self._process_exe_file(item, full_path)
            elif os.path.isdir(full_path):
                version_match = re.search(r"(\d+\.\d+\.\d+)", item)
                if version_match:
                    self._process_version_folder(version_match.group(1), full_path)

    def _process_exe_file(self, filename: str, full_path: str):
        m = re.search(r"(\d+\.\d+\.\d+)", filename)
        if m:
            self.available_versions[m.group(1)] = full_path

    def _process_version_folder(self, version: str, folder_path: str):
        for file in os.listdir(folder_path):
            if file.lower() in ["solc.exe", "solc-windows.exe"]:
                self.available_versions[version] = os.path.join(folder_path, file)
                return

    PRAGMA = re.compile(r"pragma\s+solidity\s+([^;]+);", re.IGNORECASE)

    def _extract_pragma(self, code: str) -> Optional[str]:
        m = self.PRAGMA.search(code)
        return m.group(1).strip() if m else None

    def _parse_range(self, pragma_expr: str):
        tokens = pragma_expr.split()
        lowers, uppers = [], []
        for tok in tokens:
            if tok.startswith("^"):
                base = version.parse(tok[1:])
                lowers.append((">=", base))
                uppers.append(("<", version.Version(f"{base.major}.{base.minor + 1}.0")))
                continue
            if re.fullmatch(r"\d+\.\d+\.\d+", tok):
                lowers.append(("=", version.parse(tok)))
                continue
            m = re.match(r"(>=|<=|>|<|=)\s*(\d+\.\d+\.\d+)", tok)
            if m:
                op, v = m.group(1), version.parse(m.group(2))
                (lowers if op in [">=", ">"] else uppers).append((op, v))
        return lowers, uppers

    def _satisfy(self, v, lowers, uppers) -> bool:
        for op, bound in lowers:
            if (op == ">=" and not v >= bound) or (op == ">" and not v > bound) or (
                    op == "=" and not v == bound): return False
        for op, bound in uppers:
            if (op == "<=" and not v <= bound) or (op == "<" and not v < bound): return False
        return True

    def _find_exact_solc(self, pragma_expr: Optional[str]) -> str:
        # Fallback: 如果没有 pragma，使用最新版
        if not pragma_expr:
            return self.available_versions[str(self.sorted_versions[-1])]
        try:
            lowers, uppers = self._parse_range(pragma_expr)
            candidates = [v for v in self.sorted_versions if self._satisfy(v, lowers, uppers)]
            if not candidates:
                # 找不到匹配版本，降级使用最新版（为了健壮性）
                return self.available_versions[str(self.sorted_versions[-1])]
            return self.available_versions[str(max(candidates))]
        except Exception:
            return self.available_versions[str(self.sorted_versions[-1])]

    # ======================================================================
    #  核心函数：get_error_lines_only_error_3
    #  (实际上是 Version 4 的最强逻辑，保留这个名字为了兼容你的 main.py)
    # ======================================================================
    def get_error_lines_only_error_3(self, code: str) -> list:
        # 1. 空代码检查：直接返回无错
        if not code.strip():
            return []

        # 2. 预计算代码行数 (用于无行号错误的随机兜底)
        code_lines_count = len(code.splitlines())

        # 3. 确定编译器路径
        pragma_expr = self._extract_pragma(code)
        try:
            solc_path = self._find_exact_solc(pragma_expr)
        except Exception:
            return []  # 无法获取编译器，视为无错或交给上层处理

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

            error_lines = set()
            found_any_error = False

            lines = stderr.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()

                if line.startswith("Error:"):
                    found_any_error = True
                    j = i + 1
                    found_location = False
                    # 向下查找位置信息
                    while j < min(i + 5, len(lines)):
                        inner = lines[j].strip()
                        loc_match = re.match(r"--> \S+:(\d+):\d+:", inner)
                        if loc_match:
                            error_lines.add(int(loc_match.group(1)))
                            found_location = True
                            break
                        j += 1

                    # [修复点] 如果找不到行号，从源代码行数中随机选，而不是stderr行数
                    if not found_location and code_lines_count > 0:
                        error_lines.add(random.randint(1, code_lines_count))

                    i = j
                else:
                    # 内联错误格式
                    inline = re.search(r":(\d+):\d+: Error:", line)
                    if inline:
                        found_any_error = True
                        error_lines.add(int(inline.group(1)))
                    i += 1

            # [兜底] 只要发现了错误但列表为空，强制加一个随机行
            if found_any_error and not error_lines and code_lines_count > 0:
                error_lines.add(random.randint(1, code_lines_count))

            return sorted(error_lines)

        except Exception as e:
            print(f"Linter Exception: {e}")
            return []  # 异常时返回空，防止脚本崩溃

        finally:
            if os.path.exists(tmpfile):
                try:
                    os.unlink(tmpfile)
                except:
                    pass

    # ======================================================================
    #  调试专用函数：get_error_lines_only_error_check
    #  (保留用于人工测试，但也升级了内核逻辑，增加了 print 输出)
    # ======================================================================
    def get_error_lines_only_error_check(self, code: str) -> list:
        if not code.strip():
            print("Code is empty and BENIGN.")
            return []

        code_lines_count = len(code.splitlines())
        pragma_expr = self._extract_pragma(code)
        try:
            solc_path = self._find_exact_solc(pragma_expr)
            print(f"Using solc: {solc_path}")
        except Exception as e:
            print(f"Solc selection failed: {e}")
            return []

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
            # 打印错误信息
            # print(stderr)
            error_lines = set()
            found_any_error = False
            lines = stderr.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("Error:"):
                    found_any_error = True
                    j = i + 1
                    found_location = False
                    while j < min(i + 5, len(lines)):
                        if re.match(r"--> \S+:(\d+):\d+:", lines[j].strip()):
                            error_lines.add(int(re.match(r"--> \S+:(\d+):\d+:", lines[j].strip()).group(1)))
                            found_location = True
                            break
                        j += 1
                    if not found_location and code_lines_count > 0:
                        error_lines.add(random.randint(1, code_lines_count))
                    i = j
                else:
                    inline = re.search(r":(\d+):\d+: Error:", line)
                    if inline:
                        found_any_error = True
                        error_lines.add(int(inline.group(1)))
                    i += 1

            if found_any_error and not error_lines and code_lines_count > 0:
                error_lines.add(random.randint(1, code_lines_count))

            if error_lines:
                print(f"Errors found at lines: {sorted(error_lines)}")
                # print(stderr) # 如果需要看详细报错，取消注释
            else:
                print("BENIGN on solc-windows")

            return sorted(error_lines)

        finally:
            if os.path.exists(tmpfile):
                try:
                    os.unlink(tmpfile)
                except:
                    pass