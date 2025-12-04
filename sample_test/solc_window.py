import subprocess
import tempfile
import os
import re
from typing import Dict, Optional


class SolidityLinter:
    """
    Solidity 语法错误行号提取器，支持混合版本格式。
    """

    def __init__(self, solc_versions_dir: str = r"D:\solc"):
        """
        初始化多版本 linter。
        :param solc_versions_dir: 包含多个版本 solc 的目录（已解压）
        """
        self.solc_versions_dir = solc_versions_dir
        self.available_versions: Dict[str, str] = {}
        self._discover_versions()

        if not self.available_versions:
            raise RuntimeError(f" 在目录 {solc_versions_dir} 中找不到任何 solc 可执行文件")

    def _discover_versions(self):
        """发现目录中的所有 solc 版本（支持任意命名的文件夹，只要包含 x.x.x 版本号）"""
        if not os.path.exists(self.solc_versions_dir):
            raise RuntimeError(f" solc 目录不存在: {self.solc_versions_dir}")

        for item in os.listdir(self.solc_versions_dir):
            full_path = os.path.join(self.solc_versions_dir, item)

            if os.path.isfile(full_path) and item.endswith('.exe'):
                # 处理单独的 exe 文件（如 solc-windows-v0.8.20.exe）
                self._process_exe_file(item, full_path)
            elif os.path.isdir(full_path):
                # 尝试从文件夹名中提取版本号（支持 solc-windows-0.4.25, v0.6.12 等）
                version_match = re.search(r'(\d+\.\d+\.\d+)', item)
                if version_match:
                    version_str = version_match.group(1)
                    self._process_version_folder(version_str, full_path)
                else:
                    print(f"跳过文件夹（未识别版本号）: {item}")

    def _process_exe_file(self, filename: str, full_path: str):
        """处理单个 exe 文件，从文件名提取版本"""
        version_match = re.search(r'(\d+\.\d+\.\d+)', filename)
        if version_match:
            version = version_match.group(1)
            self.available_versions[version] = full_path
            # print(f"发现 solc 版本: {version} -> {full_path}")
        else:
            print(f"警告: 无法从文件名提取版本: {filename}")

    def _process_version_folder(self, version: str, folder_path: str):
        """在版本文件夹中查找 solc 可执行文件"""
        for file_item in os.listdir(folder_path):
            if file_item.lower() in ['solc.exe', 'solc-windows.exe']:
                exe_path = os.path.join(folder_path, file_item)
                if os.path.isfile(exe_path):
                    self.available_versions[version] = exe_path
                    # print(f"发现 solc 版本: {version} -> {exe_path}")
                    return  # 找到一个即可
        print(f"警告: 文件夹 {folder_path} 中未找到 solc.exe 或 solc-windows.exe")

    def _extract_version_from_code(self, code: str) -> Optional[str]:
        """从 Solidity 代码中提取版本要求"""
        pragma_pattern = re.compile(r'pragma\s+solidity\s+([^;]+);', re.IGNORECASE | re.MULTILINE)
        match = pragma_pattern.search(code)
        if match:
            version_spec = match.group(1).strip()
            version_match = re.search(r'(\d+\.\d+\.\d+)', version_spec)
            if version_match:
                return version_match.group(1)
            major_minor_match = re.search(r'(\d+\.\d+)', version_spec)
            if major_minor_match:
                return major_minor_match.group(1) + ".0"
        return None

    def _find_best_matching_solc(self, target_version: str) -> str:
        """按大版本（major.minor）匹配最合适的 solc"""
        if not target_version:
            return self._get_latest_solc()

        target_major_minor = '.'.join(target_version.split('.')[:2])
        compatible_versions = [
            (v, p) for v, p in self.available_versions.items()
            if '.'.join(v.split('.')[:2]) == target_major_minor
        ]

        if compatible_versions:
            best_version = max(compatible_versions, key=lambda x: x[0])  # 按版本字符串排序取最大
            print(f"使用兼容版本: {target_version} -> {best_version[0]}")
            return best_version[1]

        print(f"警告: 未找到兼容 {target_version} 的 solc，使用最新版本")
        return self._get_latest_solc()

    def _get_latest_solc(self) -> str:
        """获取最新版本的 solc"""
        latest_version = max(self.available_versions.keys())
        return self.available_versions[latest_version]

    def get_error_lines(self, code: str) -> list:
        """获取语法错误行号（1-based）"""
        if not code.strip():
            return []

        target_version = self._extract_version_from_code(code)
        print(f"样本版本要求: {target_version if target_version else '未指定'}")

        solc_path = self._find_best_matching_solc(target_version)
        print(f"使用的 solc: {solc_path}")

        with tempfile.NamedTemporaryFile("w", suffix=".sol", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [solc_path, tmp_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            print(f"solc输出：\n{result.stderr}")

            error_lines = set()
            patterns = [
                r"-->\s*[^\r\n]*?:(\d+):\d+:",
                r"^(?:[^\r\n]*?):(\d+):\d+:",
                r"Error:\s*[^\r\n]*?\((\d+)\)",
            ]

            for pattern in patterns:
                for match in re.findall(pattern, result.stderr, re.MULTILINE):
                    try:
                        line_num = int(match)
                        if 1 <= line_num <= 10000:
                            error_lines.add(line_num)
                    except ValueError:
                        continue

            return sorted(error_lines)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def list_available_versions(self) -> list:
        """返回所有可用版本（排序）"""
        return sorted(self.available_versions.keys())


# 使用示例和测试
if __name__ == "__main__":
    try:
        linter = SolidityLinter()
        print(f"可用版本: {linter.list_available_versions()}")

        test_cases = [
            {
                "name": "0.4.x 版本代码",
                "code": """pragma solidity ^0.4.11;
contract IconomiBlackHole {
    function test() {
        var x = 10;  
    }
}"""
            },
            {
                "name": "0.8.x 版本代码",
                "code": """pragma solidity ^0.8.0;
contract Test {
    function foo() public {
        uint x = 10  // missing semicolon
        y = x + 1;   // undeclared variable
    }
}"""
            },
            {
                "name": "复杂版本范围",
                "code": """pragma solidity >=0.4.0 <0.6.0;
contract ComplexVersion {
    function test() public {
    }
}"""
            },
            {
                "name": "未指定版本",
                "code": """contract NoVersion {
    function test() public {
        uint x = 10;
    }
}"""
            }
        ]

        for test_case in test_cases:
            print(f"\n{'=' * 50}")
            print(f"测试: {test_case['name']}")
            print(f"{'=' * 50}")
            errors = linter.get_error_lines(test_case["code"])
            print(f"检测到的错误行号: {errors}")


    except Exception as e:
        print(f"错误: {e}")
