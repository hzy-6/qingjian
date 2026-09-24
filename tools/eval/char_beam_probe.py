#!/usr/bin/env python3
"""Offline character-LM beam proposals for complete pinyin sentences.

The frozen gold is used only for metrics.  Input syllable boundaries come from
an aligned existing candidate; character alternatives come from the lexicon.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from local_variant_ceiling import align, load_readings


def train(
    paths: list[Path],
    megabytes: int,
    order: int,
    spread: bool,
    join_lccc_tokens: bool = False,
) -> list[Counter]:
    counts = [Counter() for _ in range(order)]
    for path in paths:
        block_count = 16 if spread else 1
        block_bytes = megabytes * 1024 * 1024 // block_count
        file_size = path.stat().st_size
        with path.open("rb") as source:
            for block in range(block_count):
                offset = block * file_size // block_count
                source.seek(offset)
                if offset:
                    source.readline()  # 跳过从中间切入的半行
                remaining = block_bytes
                while remaining > 0:
                    raw = source.readline()
                    if not raw:
                        break
                    remaining -= len(raw)
                    if join_lccc_tokens and path.name.startswith("lccc"):
                        # LCCC inserts spaces at token boundaries. They are not
                        # sentence boundaries; punctuation still splits Han runs.
                        raw = raw.replace(b" ", b"")
                    prefix = ""
                    for char in raw.decode("utf-8", errors="replace"):
                        if not "\u3400" <= char <= "\u9fff":
                            prefix = ""
                            continue
                        prefix = (prefix + char)[-order:]
                        for size, table in enumerate(counts, 1):
                            if len(prefix) >= size:
                                table[prefix[-size:]] += 1
    return counts


def character_score(prefix: str, char: str, counts: list[Counter], total: int) -> float:
    vocabulary = len(counts[0])
    unigram = (counts[0][char] + 0.1) / (total + 0.1 * vocabulary)
    probabilities = [unigram]
    for order in range(2, len(counts) + 1):
        if len(prefix) < order - 1:
            break
        previous = prefix[-(order - 1):]
        probabilities.append(
            (counts[order - 1][previous + char] + 0.1)
            / (counts[order - 2][previous] + 0.1 * vocabulary)
        )
    # Favor the longest observed context, while preserving backoff for unseen
    # sequences and newly composed domain terms.
    weights = {
        3: [0.1, 0.3, 0.6],
        4: [0.05, 0.15, 0.25, 0.55],
        5: [0.03, 0.07, 0.15, 0.25, 0.5],
    }[len(counts)]
    active = weights[:len(probabilities)]
    probability = sum(a * b for a, b in zip(active, probabilities)) / sum(active)
    return math.log(probability)


def propose(
    syllables: tuple[str, ...],
    choices: dict[str, list[str]],
    counts: list[Counter],
    width: int,
    char_rank: int,
) -> list[tuple[float, str]]:
    total = sum(counts[0].values())
    beam = [(0.0, "")]
    for syllable in syllables:
        next_beam = []
        for score, prefix in beam:
            for char in choices.get(syllable, ())[:char_rank]:
                next_beam.append(
                    (score + character_score(prefix, char, counts, total), prefix + char)
                )
        next_beam.sort(reverse=True)
        beam = next_beam[:width]
        if not beam:
            return []
    return beam


def score_text(text: str, counts: list[Counter]) -> float:
    total = sum(counts[0].values())
    value = 0.0
    for at, char in enumerate(text):
        value += character_score(text[:at], char, counts, total)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("dict", type=Path, nargs="+")
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--corpus-mb", type=int, default=64)
    parser.add_argument("--beam", type=int, default=128)
    parser.add_argument("--char-rank", type=int, default=30)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument("--dump", type=Path)
    parser.add_argument("--order", type=int, choices=[3, 4, 5], default=3)
    parser.add_argument("--prefix-only", action="store_true")
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()
    readings, choices = load_readings(args.dict)
    counts = train(args.corpus, args.corpus_mb, args.order, not args.prefix_only)
    if args.min_count > 1:
        for index in range(2, len(counts)):
            counts[index] = Counter({
                gram: count for gram, count in counts[index].items()
                if count >= args.min_count
            })
    metrics = Counter()
    examples = []
    dump = args.dump.open("w", encoding="utf-8") if args.dump else None
    for line in args.jsonl.open(encoding="utf-8"):
        row = json.loads(line)
        pool = row["rerank_pool"]
        syllables = next(
            (item for text in pool for item in align(text, row["pinyin"], readings)), None
        )
        metrics["total"] += 1
        if row["gold"] in pool:
            metrics["baseline"] += 1
        if syllables is None:
            metrics["unaligned"] += 1
            continue
        scored = [
            (score, text)
            for score, text in propose(syllables, choices, counts, args.beam, args.char_rank)
            if text not in pool
        ]
        suggested = [text for _, text in scored]
        if dump:
            print(json.dumps({
                "gold": row["gold"],
                "pool": [[text, score_text(text, counts)] for text in pool],
                "proposals": [[text, score] for score, text in scored],
            }, ensure_ascii=False), file=dump)
        if row["gold"] in suggested:
            metrics["reachable_beam"] += 1
        selected = pool[: 8 - args.slots] + suggested[: args.slots]
        if row["gold"] in selected:
            metrics["selected"] += 1
        if row["gold"] in suggested[: args.slots]:
            metrics["new"] += 1
            if len(examples) < 20:
                examples.append((row["gold"], suggested.index(row["gold"]) + 1))
        if row["gold"] in pool and row["gold"] not in selected:
            metrics["lost"] += 1
    print(dict(metrics))
    print("new examples:", examples)
    if dump:
        dump.close()


if __name__ == "__main__":
    main()
