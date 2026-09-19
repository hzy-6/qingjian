#!/usr/bin/env python3
"""从 lm.qj 容器导出旧计数 TSV,再与新统计做 fill-in 混合。

用法:
  python3 tools/corpus/lm_fillin.py dump   /tmp/lm-old.qj  data/generated/old-unigram.tsv data/generated/old-bigram.tsv
  python3 tools/corpus/lm_fillin.py merge  data/generated/old-unigram.tsv data/generated/old-bigram.tsv \
         data/generated/lm-unigram.tsv data/generated/lm-bigram.tsv \
         data/generated/lm-unigram.tsv data/generated/lm-bigram.tsv
"""
import struct
import sys

# .qj 容器:Header(32B) + SectionEntry[](24B each) + 各节正文(8 字节对齐)
HEADER = struct.Struct("<8sHHI16s")
SECTION = struct.Struct("<4sIQQ")
WORD_ENTRY = struct.Struct("<IIH2x")  # text_start u32, count u32, text_len u16 (+2 填充)
SUCCESSOR = struct.Struct("<II")


def sections(data):
    magic, version, kind, count, _ = HEADER.unpack_from(data, 0)
    assert magic == b"QINGJIAN", magic
    out = {}
    for i in range(count):
        tag, _, offset, length = SECTION.unpack_from(data, HEADER.size + i * SECTION.size)
        out[tag.decode()] = (offset, length)
    return out


def dump(path, uni_out, bi_out):
    data = open(path, "rb").read()
    secs = sections(data)
    word_off, word_len = secs["WORD"]
    entr_off, entr_len = secs["ENTR"]
    offs_off, _ = secs["OFFS"]
    succ_off, _ = secs["SUCC"]
    words = data[word_off : word_off + word_len]
    entries = [WORD_ENTRY.unpack_from(data, entr_off + i * WORD_ENTRY.size) for i in range(entr_len // WORD_ENTRY.size)]
    offsets = [x[0] for x in struct.iter_unpack("<I", data[offs_off : offs_off + (len(entries) + 1) * 4])]

    def text(entry):
        start, _, length = entry
        return words[start : start + length].decode()

    with open(uni_out, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(f"{text(e)}\t{e[1]}\n")
    pairs = 0
    with open(bi_out, "w", encoding="utf-8") as f:
        for i, e in enumerate(entries):
            prev = text(e)
            for j in range(offsets[i], offsets[i + 1]):
                word_id, count = SUCCESSOR.unpack_from(data, succ_off + j * SUCCESSOR.size)
                f.write(f"{prev}\t{text(entries[word_id])}\t{count}\n")
                pairs += 1
    print(f"导出 {len(entries)} 词 / {pairs} 二元")


def read_tsv(path, cols):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= cols:
                out[tuple(parts[: cols - 1])] = int(parts[cols - 1])
    return out


def merge(old_u, old_b, new_u, new_b, out_u, out_b):
    old_uni = read_tsv(old_u, 2)
    old_bi = read_tsv(old_b, 3)
    new_uni = read_tsv(new_u, 2)
    new_bi = read_tsv(new_b, 3)
    old_total = sum(c for (w,), c in old_uni.items() if w != "<s>")
    new_total = sum(c for (w,), c in new_uni.items() if w != "<s>")
    scale = old_total / new_total
    print(f"旧总计 {old_total} / 新总计 {new_total},缩放 {scale:.4f}")

    merged_uni = dict(old_uni)
    added_words = 0
    for key, c in new_uni.items():
        if key not in merged_uni:
            merged_uni[key] = max(1, round(c * scale))
            added_words += 1

    merged_bi = dict(old_bi)
    added_pairs = 0
    for pair, c in new_bi.items():
        if pair not in merged_bi:
            (prev, _) = pair
            # 截到旧语境前词计数的一半:别让外来计数在旧分母上造出过强的条件概率
            cap = max(1, merged_uni.get((prev,), 0))
            merged_bi[pair] = min(max(1, round(c * scale)), cap)
            added_pairs += 1

    with open(out_u, "w", encoding="utf-8") as f:
        for (w,), c in merged_uni.items():
            f.write(f"{w}\t{c}\n")
    with open(out_b, "w", encoding="utf-8") as f:
        for (prev, succ), c in merged_bi.items():
            f.write(f"{prev}\t{succ}\t{c}\n")
    print(f"混合完成:一元 {len(merged_uni)}(新填 {added_words}),二元 {len(merged_bi)}(新填 {added_pairs})")


if __name__ == "__main__":
    if sys.argv[1] == "dump":
        dump(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        merge(*sys.argv[2:])
