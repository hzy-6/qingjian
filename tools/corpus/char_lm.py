#!/usr/bin/env python3
"""Stream sorted character n-gram counts for `pack_char`.

The default recipe matches the exploratory 5-gram run: first 64 MiB of each
supplied corpus, Han runs split at punctuation.  The evaluator and generator
share the same training function so score calculations can be compared exactly.
"""

import argparse
import heapq
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from char_beam_probe import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpora", type=Path, nargs="+")
    parser.add_argument("--corpus-mb", type=int, default=64)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--spread", action="store_true")
    parser.add_argument("--join-lccc-tokens", action="store_true")
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()
    counts = train(
        args.corpora, args.corpus_mb, args.order, args.spread, args.join_lccc_tokens
    )
    total = sum(counts[0].values())
    vocabulary = len(counts[0])
    out = sys.stdout.buffer
    out.write(f"#TOTAL\t{total}\n#VOCAB\t{vocabulary}\n".encode())
    ordered = []
    for index, table in enumerate(counts):
        items = (
            (gram, count)
            for gram, count in table.items()
            if index < 2 or count >= args.min_count
        )
        ordered.append(iter(sorted(items)))
    count = 0
    for gram, frequency in heapq.merge(*ordered):
        out.write(f"{gram}\t{frequency}\n".encode())
        count += 1
    print(f"streamed {count} character n-grams", file=sys.stderr)


if __name__ == "__main__":
    main()
