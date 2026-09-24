#!/usr/bin/env python3
"""分析 --eval-jsonl 中正确句未进入重排池的样本。"""

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="qingjian-cli --eval-jsonl 的输出")
    parser.add_argument("--top", type=int, default=20, help="显示多少个高频缺失字")
    args = parser.parse_args()

    total = 0
    probed = 0
    missed = 0
    absent_anywhere = 0
    absent_at_position = 0
    missing_chars: collections.Counter[str] = collections.Counter()
    nearest_distance: collections.Counter[int | str] = collections.Counter()
    examples: dict[str, tuple[str, str]] = {}
    with args.jsonl.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            total += 1
            gold = row["gold"]
            pool = row["rerank_pool"]
            if not pool:
                continue
            probed += 1
            if gold in pool:
                continue
            missed += 1
            same_length = [candidate for candidate in pool if len(candidate) == len(gold)]
            if same_length:
                nearest_distance[min(
                    sum(left != right for left, right in zip(gold, candidate))
                    for candidate in same_length
                )] += 1
            else:
                nearest_distance["不同长"] += 1
            if any(char not in "".join(pool) for char in gold):
                absent_anywhere += 1
            missing = [
                char
                for index, char in enumerate(gold)
                if not any(candidate[index] == char for candidate in same_length)
            ]
            if missing:
                absent_at_position += 1
            for char in set(missing):
                missing_chars[char] += 1
                examples.setdefault(char, (gold, pool[0]))

    print(f"评测 {total} 句；有重排池 {probed}；oracle miss {missed}")
    print(f"正确字在整池完全缺席：{absent_anywhere}/{missed}")
    print(f"正确字在对应位置缺席：{absent_at_position}/{missed}")
    print(f"与最近同长路径差几个字：{dict(sorted(nearest_distance.items(), key=lambda item: str(item[0])))}")
    print("对应位置缺失最多的字（句数、原句、池内首选）：")
    for char, count in missing_chars.most_common(args.top):
        gold, candidate = examples[char]
        print(f"{char}\t{count}\t{gold}\t{candidate}")


if __name__ == "__main__":
    main()
