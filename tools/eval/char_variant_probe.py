#!/usr/bin/env python3
"""Offline probe: rank same-pinyin one-character edits with a character n-gram LM.

Gold labels are used only by the final metrics.  This does not modify the IME.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from local_variant_ceiling import align, load_readings


def han(char: str) -> bool:
    return "\u3400" <= char <= "\u9fff"


def train(paths: list[Path], megabytes: int) -> tuple[Counter, Counter, Counter]:
    one, two, three = Counter(), Counter(), Counter()
    for path in paths:
        remaining = megabytes * 1024 * 1024
        with path.open(encoding="utf-8") as source:
            for line in source:
                remaining -= len(line.encode("utf-8"))
                if remaining < 0:
                    break
                previous = ""
                second = ""
                for char in line:
                    if not han(char):
                        previous = second = ""
                        continue
                    one[char] += 1
                    if previous:
                        two[previous + char] += 1
                    if second:
                        three[second + previous + char] += 1
                    second, previous = previous, char
    return one, two, three


def windows(text: str, at: int, n: int):
    for start in range(max(0, at - n + 1), min(at + 1, len(text) - n + 1)):
        window = text[start : start + n]
        if all(han(c) for c in window):
            yield window


def delta(text: str, variant: str, at: int, counts: tuple[Counter, Counter, Counter]) -> float:
    one, two, three = counts
    value = 0.0
    for n, table, weight in [(2, two, 1.0), (3, three, 0.7)]:
        for old, new in zip(windows(text, at, n), windows(variant, at, n)):
            # Same prefix denominator is approximated here: comparison only
            # ranks local edits in a single source sentence.
            value += weight * (math.log1p(table[new]) - math.log1p(table[old]))
    value += 0.2 * (math.log1p(one[variant[at]]) - math.log1p(one[text[at]]))
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("dict", type=Path, nargs="+")
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--corpus-mb", type=int, default=64)
    parser.add_argument("--base", type=int, default=8)
    parser.add_argument("--char-rank", type=int, default=20)
    parser.add_argument("--slots", type=int, default=3)
    args = parser.parse_args()
    readings, choices = load_readings(args.dict)
    counts = train(args.corpus, args.corpus_mb)
    print("ngrams:", [len(c) for c in counts])
    metrics = Counter()
    thresholds = [-10, -5, 0, 2, 4, 6, 8, 10, 12, 15, 20, 30]
    threshold_results = {threshold: Counter() for threshold in thresholds}
    for line in args.jsonl.open(encoding="utf-8"):
        row = json.loads(line)
        pool = row["rerank_pool"]
        proposals = {}
        for candidate in pool[: args.base]:
            for syllables in align(candidate, row["pinyin"], readings):
                for at, syllable in enumerate(syllables):
                    for char in choices.get(syllable, ())[: args.char_rank]:
                        if char == candidate[at]:
                            continue
                        variant = candidate[:at] + char + candidate[at + 1 :]
                        if variant in pool:
                            continue
                        score = delta(candidate, variant, at, counts)
                        proposals[variant] = max(score, proposals.get(variant, -math.inf))
        gold = row["gold"]
        metrics["total"] += 1
        if gold in pool:
            metrics["baseline"] += 1
        if gold in proposals:
            metrics["reachable"] += 1
        ordered = sorted(proposals, key=lambda key: proposals[key], reverse=True)
        keep = pool[: 8 - args.slots] + ordered[: args.slots]
        if gold in keep:
            metrics["selected"] += 1
        if gold in ordered[: args.slots] and gold not in pool:
            metrics["new"] += 1
        for threshold in thresholds:
            accepted = [item for item in ordered if proposals[item] > threshold][: args.slots]
            selected = pool[: 8 - len(accepted)] + accepted
            result = threshold_results[threshold]
            result["hit"] += gold in selected
            result["new"] += gold in accepted
            result["lost"] += gold in pool and gold not in selected
    print(dict(metrics))
    print("thresholds:", {key: dict(value) for key, value in threshold_results.items()})


if __name__ == "__main__":
    main()
