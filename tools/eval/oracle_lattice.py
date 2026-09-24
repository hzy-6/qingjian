#!/usr/bin/env python3
"""把整句 oracle miss 分成词库/格子截断/路径搜索三类。

读 --eval-jsonl 的冻结拼音和生成词库 TSV。诊断只考虑精确读音和词频，
不模拟模糊音、个人学习、拼写纠错或 Viterbi 分数。
"""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("dictionaries", nargs="+", type=Path, help="与评测相同的 TSV 词库")
    parser.add_argument("--span-width", type=int, default=6)
    args = parser.parse_args()

    by_word: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_key: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for path in args.dictionaries:
        with path.open(encoding="utf-8") as file:
            for line in file:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3 or fields[0].startswith("#"):
                    continue
                try:
                    frequency = int(fields[2])
                except ValueError:
                    continue
                word, spaced = fields[:2]
                by_word[word].append((spaced.replace(" ", ""), spaced))
                by_key[spaced].append((word, frequency))
    rank = {}
    for key, words in by_key.items():
        words.sort(key=lambda item: -item[1])
        for index, (word, _) in enumerate(words, 1):
            rank[(word, key)] = min(index, rank.get((word, key), index))

    def best_path(gold: str, pinyin: str) -> tuple[int, tuple[tuple[str, int], ...]]:
        @lru_cache(None)
        def visit(text_at: int, pinyin_at: int) -> tuple[int, tuple[tuple[str, int], ...]]:
            if text_at == len(gold):
                return (0, ()) if pinyin_at == len(pinyin) else (1_000_001, ())
            best = (1_000_001, ())
            for end in range(text_at + 1, min(len(gold), text_at + 8) + 1):
                word = gold[text_at:end]
                for joined, spaced in by_word.get(word, ()):
                    if not pinyin.startswith(joined, pinyin_at):
                        continue
                    rest_rank, rest_path = visit(end, pinyin_at + len(joined))
                    needed = max(rank[(word, spaced)], rest_rank)
                    if needed < best[0]:
                        best = (needed, ((word, rank[(word, spaced)]),) + rest_path)
            return best

        return visit(0, 0)

    counts: Counter[str] = Counter()
    bottlenecks: Counter[str] = Counter()
    ranks: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    with args.jsonl.open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            gold, pinyin, pool = row["gold"], row["pinyin"], row["rerank_pool"]
            if not pool or gold in pool:
                continue
            required, path = best_path(gold, pinyin)
            if required > 1_000_000:
                cause = "词库/读音不可达"
            elif required > args.span_width:
                cause = "格子前 N 候选不可达"
                band = "7–10" if required <= 10 else "11–20" if required <= 20 else "21+"
                ranks[band] += 1
                for word in {word for word, word_rank in path if word_rank == required}:
                    bottlenecks[word] += 1
            else:
                cause = "格子可达但最终池未收录"
            counts[cause] += 1
            if len(examples[cause]) < 5:
                examples[cause].append(gold)
    print(f"oracle miss: {sum(counts.values())}")
    for cause, count in counts.items():
        print(f"{cause}: {count}；例：{' / '.join(examples[cause])}")
    if bottlenecks:
        print(f"格子瓶颈词频名次：{dict(ranks)}")
        print(f"常见瓶颈词：{bottlenecks.most_common(20)}")


if __name__ == "__main__":
    main()
