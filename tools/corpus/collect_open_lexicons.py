#!/usr/bin/env python3
"""从开源词库发现青简缺词，但不直接改产品词库。

默认只保留 Jieba、Rime Essay 和 Rime Luna Pinyin 三方都有的词，
再用青简现有词库去重、用本地语料计数。输出是待审核 TSV，
不应不经验收就合并。
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import subprocess
import tempfile
from pathlib import Path


HAN_WORD = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]{2,6}$")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jieba", type=Path, required=True, help="Jieba dict.txt")
    parser.add_argument("--essay", type=Path, required=True, help="Rime Essay essay.txt")
    parser.add_argument("--luna", type=Path, required=True, help="Rime Luna Pinyin dict yaml")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--corpus", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-corpus-count", type=int, default=2)
    return parser.parse_args()


def load_known(repo: Path) -> set[str]:
    paths = [
        repo / "assets/lexicon/dict.tsv",
        repo / "assets/lexicon/domain_words.tsv",
        repo / "assets/lexicon/brand.tsv",
        *sorted((repo / "assets/lexicon/dicts").glob("*.tsv")),
    ]
    words: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as rows:
            for row in rows:
                if row.startswith("#") or not row.strip():
                    continue
                words.add(row.split("\t", 1)[0])
    return words


def load_jieba(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open(encoding="utf-8") as rows:
        for row in rows:
            fields = row.rstrip().split(" ")
            if len(fields) >= 2 and fields[1].isdigit():
                result[fields[0]] = int(fields[1])
    return result


def load_tab_frequency(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open(encoding="utf-8") as rows:
        for row in rows:
            fields = row.rstrip().split("\t")
            if len(fields) >= 2 and fields[1].isdigit():
                result[fields[0]] = int(fields[1])
    return result


def load_luna(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    in_data = False
    with path.open(encoding="utf-8") as rows:
        for row in rows:
            if row.strip() == "...":
                in_data = True
                continue
            if not in_data or row.startswith("#") or not row.strip():
                continue
            fields = row.rstrip().split("\t")
            if len(fields) >= 2:
                result.setdefault(fields[0], fields[1])
    return result


def corpus_counts(words: list[str], corpora: list[Path]) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    if not corpora:
        return counts
    with tempfile.NamedTemporaryFile("w", encoding="utf-8") as patterns:
        patterns.write("\n".join(words))
        patterns.flush()
        command = [
            "rg", "--no-filename", "--only-matching", "--fixed-strings",
            "--file", patterns.name, *map(str, corpora),
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        assert process.stdout is not None
        for match in process.stdout:
            counts[match.rstrip("\n")] += 1
        status = process.wait()
        if status not in (0, 1):
            raise subprocess.CalledProcessError(status, command)
    return counts


def main() -> None:
    args = arguments()
    known = load_known(args.repo)
    jieba = load_jieba(args.jieba)
    essay = load_tab_frequency(args.essay)
    luna = load_luna(args.luna)

    words = sorted(
        word for word in luna
        if word not in known and HAN_WORD.fullmatch(word) and word in jieba and word in essay
    )
    eligible_count = len(words)
    counts = corpus_counts(words, args.corpus)
    words = [word for word in words if counts[word] >= args.min_corpus_count]
    words.sort(key=lambda word: (counts[word], jieba[word], essay[word], word), reverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        writer.writerow(["# word", "pinyin_upstream", "corpus_hits", "jieba_freq", "essay_weight"])
        for word in words:
            writer.writerow([word, luna[word], counts[word], jieba[word], essay[word]])

    print(
        f"known={len(known)} three_source_missing={eligible_count} "
        f"corpus_confirmed={len(words)} output={args.output}"
    )


if __name__ == "__main__":
    main()
