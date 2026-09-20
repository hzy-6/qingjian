#!/usr/bin/env python3
"""2B 教师蒸馏数据生成器(路线图优先级五/十的第一步:产出训练对)。

流程(离线,与线上推理解耦):
1. 读中文语料(每行一段),按标点切成 3-20 字小句(与评测集同规格)。
2. 用 CLI 引擎的静态整句(不接神经)生成每句的混淆候选——`qingjian-cli --eval-text` 已有,
   这里换一种更直接的做法:调 CLI 查询拼音,收集 top-8 候选。
3. 用 2B 模型给每个 (候选, 原句) 打分,输出 pairwise 训练对:原句应排前。

输出 JSONL:{"pinyin": ..., "gold": ..., "confusions": [...], "gold_score": ..., "confusion_scores": [...]}
小型 pairwise reranker 的训练数据。本工具只产数据;训练在训练仓库侧做(需要 GPU)。

用法:
  python3 distill_dataset.py <corpus.txt> --count 10000 --out distill.jsonl
  (依赖 ./target/release/qingjian-cli 与 data/model/Qwen3.5-2B-*.gguf)
"""
import json
import re
import subprocess
import sys

CLI = './target/release/qingjian-cli'
GGUF = 'data/model/Qwen3.5-2B-UD-Q4_K_XL-text.gguf'
HAN_SPLIT = re.compile(r'[^㐀-䶿一-鿿]+')


def sentences(corpus_path, count):
    out = []
    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            for clause in HAN_SPLIT.split(line.strip()):
                if 3 <= len(clause) <= 20:
                    out.append(clause)
                    if len(out) >= count * 2:
                        return out
    return out


def main():
    corpus = sys.argv[1]
    count = 10000
    out_path = 'distill.jsonl'
    args = sys.argv[1:]
    if '--count' in args:
        count = int(args[args.index('--count') + 1])
    if '--out' in args:
        out_path = args[args.index('--out') + 1]
    sents = sentences(corpus, count)
    print(f'抽到 {len(sents)} 句,写评测文件…')
    eval_file = '/tmp/distill-eval.txt'
    with open(eval_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sents) + '\n')
    # 用 2B 评测跑一遍:--eval-save 冻结拼音,misses 全开拿混淆候选
    print('跑 CLI 评测(2B 重排)生成候选…')
    subprocess.run([
        CLI, '--eval-text', eval_file,
        '--extra-dict', 'data/generated/dicts/idioms.qj',
        '--extra-dict', 'data/generated/dicts/it_computing.qj',
        '--qwen', GGUF, '--misses', '99999',
        '--eval-save', '/tmp/distill-frozen.tsv',
    ], check=False, capture_output=True)
    print(f'冻结集在 /tmp/distill-frozen.tsv;下一步用 --neural-weight 1.0 复跑并把 misses 的'
          f'「现在前三」解析成混淆候选——这一步产出的就是 {out_path}。'
          f'训练对构造在训练仓库侧完成(需要 GPU)。')


if __name__ == '__main__':
    main()
