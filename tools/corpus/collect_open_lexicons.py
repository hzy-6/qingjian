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
import json
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
    parser.add_argument(
        "--input-log", type=Path,
        help="本机 input-log.jsonl；只统计未撤销的 word 上屏，不把日志原文写入输出",
    )
    parser.add_argument(
        "--min-log-commits", type=int, default=0,
        help="至少有多少次真实词上屏；默认 0 保留全部候选",
    )
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


def committed_word_counts(path: Path) -> collections.Counter[str]:
    """只把最终未撤销的词候选算作用户需求；会话重启后 id 从头计数。"""
    committed: dict[tuple[int, int], str] = {}
    retracted: set[tuple[int, int]] = set()
    session = 0
    with path.open(encoding="utf-8") as rows:
        for row in rows:
            try:
                item = json.loads(row)
            except json.JSONDecodeError:
                continue
            event = item.get("event")
            if event == "session":
                session += 1
            elif event == "commit" and item.get("source") == "word":
                if isinstance(item.get("id"), int) and isinstance(item.get("text"), str):
                    committed[(session, item["id"])] = item["text"]
            elif event == "retract" and isinstance(item.get("of"), int):
                retracted.add((session, item["of"]))
    return collections.Counter(
        word for key, word in committed.items() if key not in retracted
    )


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
    log_counts = committed_word_counts(args.input_log) if args.input_log else collections.Counter()
    words = [word for word in words if counts[word] >= args.min_corpus_count]
    words = [word for word in words if log_counts[word] >= args.min_log_commits]
    words.sort(
        key=lambda word: (log_counts[word], counts[word], jieba[word], essay[word], word),
        reverse=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "# word", "pinyin_upstream", "local_word_commits", "corpus_hits",
            "jieba_freq", "essay_weight",
        ])
        for word in words:
            writer.writerow([
                word, luna[word], log_counts[word], counts[word], jieba[word], essay[word],
            ])

    print(
        f"known={len(known)} three_source_missing={eligible_count} "
        f"corpus_confirmed={len(words)} local_word_commits={sum(log_counts[word] for word in words)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
