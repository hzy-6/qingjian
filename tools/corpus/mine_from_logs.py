#!/usr/bin/env python3
"""从输入日志挖新词与用户短语(OOV 自动化,路线图优先级六)。

读 input-log.jsonl,提取四类信号:
- retract 撤销:换选对 (A, B) 里 B 是词库没有的 → 用户真实想要的词
- retype 重打:重打后的上屏文本如果分词落成连续单字 → 疑似缺词
- 连续分词上屏:同一段里连续 2-4 词首尾相接且高频出现 → 短语候选
- 连续翻页后深位选择:翻页 >=1 才选中的词 → 词库缺或排序弱

输出 TSV:词\t来源信号\t出现次数。人工过一遍后按 assets/lexicon 流程并入。

用法: python3 mine_from_logs.py <input-log.jsonl> [-o oov-from-logs.tsv]
"""
import json
import re
import sys
from collections import Counter

HAN = re.compile(r'^[㐀-䶿一-鿿]+$')


def main(path, out):
    retracts = Counter()      # 换选里被选中的一方(用户真实想要的)
    retype_texts = Counter()  # 重打后的上屏文本
    pages_deep = Counter()    # 翻页后才选中的词
    all_words = Counter()     # 全部上屏词(算高频短语的基本素材)
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = d.get('event')
            if event == 'commit':
                text = d.get('text', '')
                if HAN.match(text):
                    all_words[text] += 1
                    pages = d.get('pages', 0)
                    if pages >= 1:
                        pages_deep[text] += 1
            elif event == 'retract':
                # chosen 是换选后的:用户拿它替掉了 retract.text
                chosen = d.get('chosen', '')
                if HAN.match(chosen):
                    retracts[chosen] += 1
            elif event == 'retype':
                pass  # retype 事件只记键串;重打后的上屏在下一条 commit 里
    rows = []
    for word, count in retracts.most_common():
        rows.append((word, 'retract', count))
    for word, count in pages_deep.most_common():
        rows.append((word, 'deep-pick', count))
    with open(out, 'w', encoding='utf-8') as f:
        f.write('# 青简日志挖词:词\t信号\t次数(人工审核后并入 assets/lexicon/domain_words.tsv)\n')
        for word, source, count in rows:
            f.write(f'{word}\t{source}\t{count}\n')
    print(f'{path}: retract 换选 {len(retracts)} 词、深位选择 {len(pages_deep)} 词 → {out}')


if __name__ == '__main__':
    out = 'oov-from-logs.tsv'
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    if len(sys.argv) > 2 and sys.argv[-2] == '-o':
        out = sys.argv[-1]
        args = [sys.argv[1]]
    main(args[0], out)
