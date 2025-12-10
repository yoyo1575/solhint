
# 抽样合约solc检查语法
python sample_jsonl.py
    "D:\paper\Synthtic_edit_sequence\lintseq\solhint\raw_sol\merge_clean_sol.jsonl" 3
    "D:\paper\Synthtic_edit_sequence\lintseq\solhint\raw_sol\clean_sol_0.jsonl" 3

python sample_jsonl.py     "D:\paper\Synthtic_edit_sequence\lintseq\solhint\raw_sol\merge_clean_sol.jsonl" 2


# test special lines
python test_single_line.py "D:\paper\Synthtic_edit_sequence\lintseq\solhint\raw_sol\merge_clean_sol.jsonl" 5383


# warning NOT error
3798

# collect_errors
python collect_errors.py "D:\paper\Synthtic_edit_sequence\lintseq\solhint\raw_sol\merge_clean_sol.jsonl"