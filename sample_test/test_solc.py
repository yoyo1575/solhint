import subprocess
import tempfile
import os
import shutil
import re


class SolidityLinter:
    """
    Solidity 语法错误行号提取器，基于 solc-windows.exe。
    用于 LintSeq 的 backward sampling：返回导致编译失败的行号列表（1-based）。
    """

    def __init__(self, solc_path=None):
        """
        初始化 linter。
        :param solc_path: solc-windows.exe 的路径，若为 None 则自动查找。
        """
        self.solc_path = solc_path or self._find_solc()
        if not os.path.isfile(self.solc_path):
            raise RuntimeError(f"❌ 找不到 solc 可执行文件: {self.solc_path}")

    @staticmethod
    def _find_solc():
        """自动查找 solc 路径"""
        candidates = [
            r"D:\solc\solc-windows.exe",
            r"D:\solc\solc.exe",
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        # 尝试从系统 PATH 查找
        path_in_env = shutil.which("solc")
        return path_in_env if path_in_env else candidates[0]

    def get_error_lines(self, code: str) -> list:
        if not code.strip():
            return []

        # 加载源代码时跳过源代码pragma版本
        filtered_lines = []
        pragma_pattern = re.compile(r'^\s*pragma\s+solidity', re.IGNORECASE)
        for line in code.splitlines(keepends=True):
            if not pragma_pattern.match(line):
                filtered_lines.append(line)

        filtered_code = ''.join(filtered_lines)

        # fake pragma solidity头，跳过solc0.8.30检查
        # 统一版本号
        fake_pragma = "pragma solidity ^0.8.0;\n"
        final_code = fake_pragma + filtered_code

        # 写入临时文件
        with tempfile.NamedTemporaryFile("w", suffix=".sol", delete=False, encoding="utf-8") as f:
            f.write(final_code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [self.solc_path, tmp_path],
                capture_output=True,
                text=True
            )

            print(f"错误信息：\n{result.stderr}")

            error_lines = set()
            pattern = r"-->\s*[^\r\n]*?:(\d+):\d+:"
            for match in re.findall(pattern, result.stderr):
                try:
                    line_num = int(match)
                    if 1 <= line_num <= 10000:  # 合理行号范围
                        error_lines.add(line_num)
                except ValueError:
                    continue

            return sorted(error_lines)
        finally:
            # 删除临时文件
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    # 创建 linter 实例
    linter = SolidityLinter(solc_path=r"D:\solc\solc-windows-0.8.30.exe")

    # 测试代码
    test_code = """
pragma solidity ^0.8.0;
contract Test {
    function foo() public {
        uint x = 10  // missing semicolon
        y = x + 1;   // undeclared variable
    }
}
"""
    test_code_1 = "pragma solidity ^0.4.11;\ncontract IconomiBlackHole {\n}"

    errors = linter.get_error_lines(test_code)
    print("错误行号:", errors)  # 例如 [5, 6]
