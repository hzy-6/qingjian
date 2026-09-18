#!/usr/bin/env python3
"""候选词表并入校验管线（生活闲聊词库类）。

用法: python3 tools/lexicon-merge-check.py <候选.tsv>   # 词\t无空格或分音节拼音\t次数
输出: /tmp/lexicon-merge-accepted.json（可并入）与复核清单（需人工）。
校验: 贪心最长音节切分 + 单字读音反查 + 词库去重。ü 一律写作 v。
"""
import json
import sys

DICT = 'assets/lexicon/dict.tsv'
DOMAIN = 'assets/lexicon/domain_words.tsv'


def load():
    words = set()
    char_readings = {}
    syllables = set()
    for line in open(DICT, encoding='utf-8'):
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 2:
            words.add(parts[0])
            sl = parts[1].split()
            syllables.update(sl)
            if len(sl) == 1 and len(parts[0]) == 1:
                char_readings.setdefault(parts[0], set()).add(sl[0])
    for line in open(DOMAIN, encoding='utf-8'):
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 3:
            words.add(parts[0])
            # 只收音节形 token，防列序错乱的行把次数/拼音串吸进音节集
            for s in parts[2].split():
                if s.isalpha():
                    syllables.add(s)
    return words, char_readings, syllables


def greedy(py, syllables):
    out, i = [], 0
    while i < len(py):
        for l in range(min(6, len(py) - i), 0, -1):
            if py[i:i + l] in syllables:
                out.append(py[i:i + l])
                i += l
                break
        else:
            return None
    return out


def main():
    path = sys.argv[1]
    words, char_readings, syllables = load()
    accepted, flagged, skipped = [], [], []
    seen = set()
    for line in open(path, encoding='utf-8'):
        parts = line.rstrip('\n').split('\t')
        if len(parts) != 3:
            skipped.append(('格式', line.strip()))
            continue
        word, py, cnt = parts
        if word in words or word in seen:
            skipped.append(('已在词库', word))
            continue
        seen.add(word)
        if ' ' in py:
            cand = py.split()
            ss = cand if all(c in syllables for c in cand) else greedy(py.replace(' ', ''), syllables)
        else:
            ss = greedy(py, syllables)
        if ss is None or len(ss) != len(word):
            flagged.append((word, py, '切分不过或字数不合'))
            continue
        bad = None
        for ch, s in zip(word, ss):
            readings = char_readings.get(ch)
            if readings and s not in readings:
                bad = (ch, s, sorted(readings)[:4])
                break
        if bad:
            flagged.append((word, py, f"字音不符 {bad}"))
            continue
        accepted.append((word, ' '.join(ss), cnt))
    # 复核处置闭环：每条 flagged 都要有处置记录（修/拒+理由），写到 disposition 文件防蒸发
    try:
        known = json.load(open(path + '.disposition.json', encoding='utf-8'))
    except Exception:
        known = {}
    pending = [f for f in flagged if f[0] not in known]
    json.dump(flagged, open(path + '.flagged.json', 'w'), ensure_ascii=False)
    if pending:
        print(f"  ⚠ {len(pending)} 条复核项未处置（写入 {path}.flagged.json，处置后同步 disposition）：")
        for f in pending[:10]:
            print('   ', f)
    print(f"可并入 {len(accepted)}，需人工 {len(flagged)}，跳过 {len(skipped)}")
    for f in flagged[:30]:
        print('  复核:', f)
    if len(flagged) > 30:
        print(f'  …共 {len(flagged)} 条')


if __name__ == '__main__':
    main()
