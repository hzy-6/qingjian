#!/usr/bin/env python3
"""Measure the oracle ceiling of one same-pinyin character edit around an IME pool.

This is an offline diagnostic: gold is used only to check coverage, never to
construct a variant.  The input pinyin is aligned with each pool sentence using
single-character dictionary readings.  Ambiguous alignments are all considered.
"""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path


def load_readings(paths: list[Path]) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    readings: dict[str, set[str]] = defaultdict(set)
    by_syllable: dict[str, dict[str, int]] = defaultdict(dict)
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3 or len(fields[0]) != 1:
                    continue
                try:
                    frequency = int(fields[2])
                except ValueError:
                    continue
                char, syllable = fields[:2]
                if " " in syllable or not syllable.isalpha():
                    continue
                readings[char].add(syllable)
                by_syllable[syllable][char] = max(
                    frequency, by_syllable[syllable].get(char, 0)
                )
    choices = {
        syllable: [char for char, _ in sorted(words.items(), key=lambda item: -item[1])]
        for syllable, words in by_syllable.items()
    }
    return readings, choices


def align(text: str, pinyin: str, readings: dict[str, set[str]]) -> tuple[tuple[str, ...], ...]:
    @lru_cache(None)
    def suffix(index: int, offset: int) -> tuple[tuple[str, ...], ...]:
        if index == len(text):
            return ((),) if offset == len(pinyin) else ()
        result = []
        for syllable in readings.get(text[index], ()):
            if pinyin.startswith(syllable, offset):
                result.extend(
                    (syllable, *rest)
                    for rest in suffix(index + 1, offset + len(syllable))
                )
        return tuple(result)

    return suffix(0, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("dictionaries", type=Path, nargs="+")
    parser.add_argument("--rank", type=int, default=20)
    args = parser.parse_args()
    readings, choices = load_readings(args.dictionaries)
    counts = Counter()
    improvements = []
    with args.jsonl.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            pool = row["rerank_pool"]
            gold = row["gold"]
            if gold in pool:
                counts["already_in_pool"] += 1
                continue
            counts["miss"] += 1
            found = None
            aligned_any = False
            for pool_rank, candidate in enumerate(pool, 1):
                if len(candidate) != len(gold):
                    continue
                differences = [i for i, (a, b) in enumerate(zip(candidate, gold)) if a != b]
                if len(differences) != 1:
                    continue
                position = differences[0]
                for syllables in align(candidate, row["pinyin"], readings):
                    aligned_any = True
                    alternatives = choices.get(syllables[position], ())[: args.rank]
                    if gold[position] in alternatives:
                        alt_rank = alternatives.index(gold[position]) + 1
                        found = (pool_rank, alt_rank, position, syllables[position])
                        break
                if found:
                    break
            if found:
                counts["one_edit_reachable"] += 1
                improvements.append((gold, *found))
                counts[f"pool_top_5_{found[0] <= 5}"] += 1
            elif aligned_any:
                counts["one_edit_bad_reading_or_rank"] += 1
            else:
                counts["multiple_edits_or_unaligned"] += 1
    print(dict(counts))
    print("one-edit pool ranks:", dict(Counter(x[1] for x in improvements)))
    print("one-edit char rank bands:", dict(Counter(
        "1-6" if x[2] <= 6 else "7-10" if x[2] <= 10 else "11-20"
        for x in improvements
    )))
    print("examples:", improvements[:20])


if __name__ == "__main__":
    main()
