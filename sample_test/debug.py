from solc_windows import SolidityLinter

print("=== 开始检测 D:\\solc 下所有 solc 可执行文件 ===")

linter = SolidityLinter()

print("\n=== 发现的 solc 版本如下：")
for v, path in sorted(linter.available_versions.items()):
    print(f"{v:<10} -> {path}")

print("\n=== 所有解析出的版本（排序后） ===")
print(linter.sorted_versions)
