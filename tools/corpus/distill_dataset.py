#!/usr/bin/env python3
"""用青简 2B 教师生成可直接训练的候选对 JSONL。

CLI 只加载一次模型，完成语料切句、拼音冻结和教师重排，并通过 ``--eval-jsonl``
输出机器可读候选。本脚本再转换为训练格式：positive 与每个 negative 构成 pairwise 训练对。
正确句没被召回的记录也保留（teacher_rank=null），供召回模型分析。
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=Path("distill.jsonl"))
    parser.add_argument("--cli", type=Path, default=Path("target/release/qingjian-cli"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/model/Qwen3.5-2B-UD-Q4_K_XL-text.gguf"),
    )
    return parser.parse_args()


def require_file(path, label):
    if not path.is_file():
        raise SystemExit(f"{label}不存在: {path}")


def main():
    args = parse_args()
    require_file(args.corpus, "语料")
    require_file(args.cli, "CLI（先 cargo build --release -p qingjian-cli --features qwen）")
    require_file(args.model, "2B GGUF")
    if args.count <= 0:
        raise SystemExit("--count 必须大于 0")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qingjian-distill-") as scratch:
        raw = Path(scratch) / "teacher.jsonl"
        frozen = Path(scratch) / "frozen.tsv"
        command = [
            str(args.cli),
            "--eval-text", str(args.corpus),
            "--eval-save", str(frozen),
            "--eval-jsonl", str(raw),
            "--qwen", str(args.model),
            "--misses", "0",
        ]
        for extra in (
            Path("data/generated/dicts/idioms.qj"),
            Path("data/generated/dicts/it_computing.qj"),
        ):
            if extra.is_file():
                command.extend(("--extra-dict", str(extra)))
        subprocess.run(command, check=True)

        written = 0
        recalled = 0
        with raw.open(encoding="utf-8") as source, args.out.open("w", encoding="utf-8") as target:
            for line in source:
                record = json.loads(line)
                gold = record["gold"]
                negatives = [text for text in record["candidates"] if text != gold]
                rank = record["gold_rank"]
                if rank is not None:
                    recalled += 1
                target.write(json.dumps({
                    "pinyin": record["pinyin"],
                    "context": record["context"],
                    "positive": gold,
                    "negatives": negatives,
                    "teacher_rank": rank,
                    "rerank_pool": record["rerank_pool"],
                }, ensure_ascii=False) + "\n")
                written += 1
                if written >= args.count:
                    break
    if written == 0:
        raise SystemExit("教师没有产出任何可用记录")
    print(f"写出 {written} 条到 {args.out}; 原句召回 {recalled}/{written}")


if __name__ == "__main__":
    main()
