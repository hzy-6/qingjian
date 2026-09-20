#!/usr/bin/env python3
"""从输入日志挖不在产品词库里的词与 2–4 段用户短语。

信号包括：撤销后换选、退格重打后的下一次上屏、翻页选择、同一应用内连续上屏短语。
输出仍需人工审核；``--dict`` 可重复指定 TSV 词库，缺省读取产品基础词库与源码词表。
"""

import argparse
import json
import re
from collections import Counter, deque
from pathlib import Path

HAN = re.compile(r"^[㐀-䶿一-鿿]+$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=Path("oov-from-logs.tsv"))
    parser.add_argument("--dict", action="append", type=Path, default=[])
    parser.add_argument("--phrase-min", type=int, default=2)
    return parser.parse_args()


def load_known(paths):
    known = set()
    for path in paths:
        if not path.is_file() or path.suffix == ".qj":
            continue
        with path.open(encoding="utf-8", errors="replace") as source:
            for line in source:
                if line.strip() and not line.startswith("#"):
                    word = line.split("\t", 1)[0].strip()
                    if word:
                        known.add(word)
    return known


def main():
    args = parse_args()
    if not args.log.is_file():
        raise SystemExit(f"日志不存在: {args.log}")
    dicts = args.dict or [
        Path("assets/lexicon/dict.tsv"),
        Path("assets/lexicon/domain_words.tsv"),
        Path("assets/lexicon/brand.tsv"),
    ]
    known = load_known(dicts)
    signals = Counter()
    pending_retype = False
    recent = deque(maxlen=4)
    current_app = None

    def add(word, source):
        if HAN.fullmatch(word) and word not in known:
            signals[(word, source)] += 1

    with args.log.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("event")
            if kind == "break":
                recent.clear()
                current_app = event.get("app")
                pending_retype = False
            elif kind == "retype":
                pending_retype = True
            elif kind == "retract":
                add(event.get("chosen", ""), "retract")
            elif kind == "commit":
                word = event.get("text", "")
                app = event.get("app")
                if app != current_app:
                    recent.clear()
                    current_app = app
                if not HAN.fullmatch(word):
                    recent.clear()
                    pending_retype = False
                    continue
                if pending_retype:
                    add(word, "retype")
                    pending_retype = False
                if event.get("pages", 0) >= 1:
                    add(word, "deep-pick")
                recent.append(word)
                words = list(recent)
                for width in range(2, min(4, len(words)) + 1):
                    phrase = "".join(words[-width:])
                    if len(phrase) <= 16 and phrase not in known:
                        signals[(phrase, f"phrase-{width}")] += 1

    rows = [
        (word, source, count)
        for (word, source), count in signals.items()
        if not source.startswith("phrase-") or count >= args.phrase_min
    ]
    rows.sort(key=lambda row: (-row[2], row[0], row[1]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as target:
        target.write("# 词\t信号\t次数（已过滤产品词库，仍需人工审核）\n")
        for word, source, count in rows:
            target.write(f"{word}\t{source}\t{count}\n")
    print(f"已知词 {len(known)}，挖出 {len(rows)} 条 OOV/短语信号 → {args.out}")


if __name__ == "__main__":
    main()
