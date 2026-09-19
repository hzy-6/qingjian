#!/usr/bin/env python3
"""把 Qwen3.5 GGUF 的词嵌入裁成纯中文输入法需要的行。

保留 id 0..KEEP_INCLUSIVE(全部 CJK/拉丁/符号),之后的行(多语言碎片、图像/音频/tts/pad 专用
token)整段裁掉;token_embd 按行删块(Q8_0 每行 hidden/32 个 34 字节块),词表 / 类型 /
eos 元数据同步重写,白名单里的特殊 token 追加到词表末尾。

用法: python3 trim_gguf_vocab.py <in.gguf> <out.gguf> <keep_inclusive> [白名单token …]
"""
import io
import os
import struct
import sys

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def read_str(f):
    n, = struct.unpack('<Q', f.read(8))
    return f.read(n)


def read_val(f, t):
    if t == 8:
        return read_str(f)
    if t == 9:
        et, n = struct.unpack('<IQ', f.read(12))
        if et == 8:
            return [read_str(f) for _ in range(n)]
        return list(struct.unpack('<' + FMT[et] * n, f.read(SZ[et] * n)))
    return struct.unpack('<' + FMT[t], f.read(SZ[t]))[0]


def val_type(v):
    if isinstance(v, (str, bytes)):
        return 8
    if isinstance(v, bool):
        return 7
    if isinstance(v, float):
        return 6
    return 4  # 一律按 u32 写(本文件里的整数数组都非负、值域 32 位内)


def write_str(out, s):
    b = s if isinstance(s, bytes) else s.encode('utf-8')
    out.write(struct.pack('<Q', len(b)) + b)


def write_kv(out, t, v):
    if t == 8:
        write_str(out, v)
    elif t == 9:
        et = val_type(v[0])
        out.write(struct.pack('<IQ', et, len(v)))
        if et == 8:
            for s in v:
                write_str(out, s)
        else:
            for x in v:
                out.write(struct.pack(FMT[et], x))
    else:
        out.write(struct.pack(FMT[t], v))


def main(src, dst, keep, whitelist):
    f = open(src, 'rb')
    magic, version, n_tensors, n_kv = struct.unpack('<4sIQQ', f.read(24))
    kvs = []
    for _ in range(n_kv):
        key = read_str(f).decode()
        t, = struct.unpack('<I', f.read(4))
        kvs.append([key, t, read_val(f, t)])
    tensors = []
    for _ in range(n_tensors):
        name = read_str(f).decode()
        nd, = struct.unpack('<I', f.read(4))
        dims = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
        tt, off = struct.unpack('<IQ', f.read(12))
        tensors.append([name, tt, list(dims), off])
    data_start = (f.tell() + 31) & ~31
    fsize = os.path.getsize(src)
    # 相邻偏移求每个张量的字节数(偏移按列表序单调,已验证)
    sizes = []
    for i, (_, _, _, off) in enumerate(tensors):
        end = tensors[i + 1][3] if i + 1 < len(tensors) else fsize - data_start
        sizes.append(end - off)
    assert sum(sizes) < fsize, "张量尺寸总和异常"

    tokens = next(v for k, _, v in kvs if k == 'tokenizer.ggml.tokens')
    token_type = next(v for k, _, v in kvs if k == 'tokenizer.ggml.token_type')
    keep_ids = list(range(keep + 1))
    id_map = {old: new for new, old in enumerate(keep_ids)}
    for i, s in enumerate(tokens):
        if i > keep and s in whitelist and i not in id_map:
            id_map[i] = len(id_map)
            keep_ids.append(i)
    new_tokens = [tokens[i] for i in keep_ids]
    for kv in kvs:
        key, t, v = kv
        if key == 'tokenizer.ggml.tokens':
            kv[2] = new_tokens
        elif key == 'tokenizer.ggml.token_type':
            kv[2] = [token_type[i] for i in keep_ids]
        elif key.endswith('.vocab_size'):
            kv[2] = len(new_tokens)
        elif key in ('tokenizer.ggml.eos_token_id', 'tokenizer.ggml.bos_token_id',
                     'tokenizer.ggml.padding_token_id', 'tokenizer.ggml.sep_token_id'):
            kv[2] = id_map.get(v, v)

    embd_idx = next(i for i, t in enumerate(tensors) if t[0] == 'token_embd.weight')
    hidden, vocab = tensors[embd_idx][2][0], tensors[embd_idx][2][1]
    assert vocab == len(tokens), f"词表维 {vocab} 与 tokens {len(tokens)} 不符"
    # 行宽按实测:张量总字节 ÷ 词表行数(对任意量化布局都成立;Q8_0 恰为 hidden/32×34,
    # UD 混合量化的 embd 不是 Q8_0,按类型猜会错——0.8B 的 Q8 版与 2B 的 UD-Q4 版都用这条)
    embd_size = sizes[embd_idx]
    assert embd_size % vocab == 0, f"词嵌入字节数 {embd_size} 不能被 {vocab} 行整除"
    row_bytes = embd_size // vocab
    tensors[embd_idx][2] = [hidden, len(new_tokens)]
    embd_bytes = len(new_tokens) * row_bytes

    out = open(dst, 'wb')
    out.write(struct.pack('<4sIQQ', magic, version, n_tensors, len(kvs)))
    for key, t, v in kvs:
        write_str(out, key)
        out.write(struct.pack('<I', t))
        write_kv(out, t, v)
    meta_end = out.tell()

    def build_infos(offsets):
        buf = io.BytesIO()
        for (name, tt, dims, _), off in zip(tensors, offsets):
            write_str(buf, name)
            buf.write(struct.pack('<I', len(dims)))
            for d in dims:
                buf.write(struct.pack('<Q', d))
            buf.write(struct.pack('<IQ', tt, off))
        return buf.getvalue()

    placeholder = build_infos([0] * n_tensors)
    pad = (-(meta_end + len(placeholder))) % 32
    data_start_out = meta_end + len(placeholder) + pad
    cur = data_start_out
    offsets = []
    absolutes = []
    for (name, _, _, _), size in zip(tensors, sizes):
        absolutes.append(cur)
        offsets.append(cur - data_start_out)  # GGUF 的 offset 相对数据区起点
        cur += embd_bytes if name == 'token_embd.weight' else size
        cur += (-cur) % 32
    out.write(build_infos(offsets))
    out.write(b'\0' * pad)
    for (name, _, _, off), size, ro in zip(tensors, sizes, absolutes):
        gap = ro - out.tell()
        assert 0 <= gap < 4096, f"{name} 对齐间隙异常 {gap}"
        out.write(b'\0' * gap)
        if name == 'token_embd.weight':
            base = data_start + off
            for row in keep_ids:
                f.seek(base + row * row_bytes)
                out.write(f.read(row_bytes))
        else:
            f.seek(data_start + off)
            remaining = size
            while remaining > 0:
                chunk = f.read(min(1 << 24, remaining))
                out.write(chunk)
                remaining -= len(chunk)
        assert out.tell() <= ((ro + (embd_bytes if name == 'token_embd.weight' else size)) + 4096), name
        if out.tell() > 2_000_000_000:
            raise SystemExit("保险丝:输出超 2GB,中止")
    print(f"完成:词表 {len(tokens)} → {len(new_tokens)}(裁 {len(tokens) - len(new_tokens)} 行,"
          f"省 {(len(tokens) - len(new_tokens)) * row_bytes / 1e6:.0f}MB),"
          f"输出 {out.tell() / 1e6:.0f}MB(原 {fsize / 1e6:.0f}MB)")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), set(sys.argv[4:]))
