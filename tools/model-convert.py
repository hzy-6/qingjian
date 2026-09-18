#!/usr/bin/env python3
"""把 HuggingFace 上的中文 GPT-2 字级模型转制为青简整句模型三件套。

用法：python3 tools/model-convert.py <模型目录或 HF id> <输出目录>
  模型目录需含 config.json / vocab.json（或 vocab.txt）/ model.safetensors。
  输出三件套 config.json / vocab.json / model.safetensors 可用
  `cargo run --release -p qingjian-dict-convert -- pack model` 打成 model.qjm。

转换内容：
- byte-level BPE 字形空间经 bytes_to_unicode 逆映射还原真实字符，只收单字 token，
  按 BERT 词表行序（char-level）或原表序（BPE）重排；特殊位 0=pad、1=unk、2=eos；
- wte 行按新词表重排；不共享输出头的模型把 lm_head 重映射为 head.weight；
- HF GPT2 张量名映射到青简命名，Conv1D 的 (in, out) 转置为 candle 的 (out, in)；f32 → f16。
"""
import json
import os
import struct
import sys

import numpy as np


def bytes_to_unicode_reverse():
    bs = list(range(ord('!'), ord('~') + 1))
    bs += list(range(ord('¡'), ord('¬') + 1))
    bs += list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def read_safetensors(path):
    data = open(path, 'rb').read()
    n = struct.unpack('<Q', data[:8])[0]
    header = json.loads(data[8:8 + n])
    base = 8 + n
    out = {}
    for name, meta in header.items():
        if name == '__metadata__':
            continue
        a, b = meta['data_offsets']
        dt = np.float16 if meta['dtype'] == 'F16' else np.float32
        out[name] = np.frombuffer(data[base + a:base + b], dtype=dt).reshape(meta['shape'])
    return out


def load_vocab(model_dir):
    """返回 (token→真实文本, 是否字符级词表)。"""
    if os.path.exists(os.path.join(model_dir, 'vocab.json')):
        raw = json.load(open(os.path.join(model_dir, 'vocab.json'), encoding='utf-8'))
        rev = bytes_to_unicode_reverse()
        decoded = {}
        for tok in raw:
            try:
                decoded[tok] = bytes(rev[c] for c in tok).decode('utf-8')
            except Exception:
                decoded[tok] = None
        return decoded, True
    if os.path.exists(os.path.join(model_dir, 'vocab.txt')):
        lines = open(os.path.join(model_dir, 'vocab.txt'), encoding='utf-8').read().split('\n')
        return {tok: tok for tok in lines if tok}, False
    raise SystemExit('找不到 vocab.json（byte-level BPE）或 vocab.txt（字符级）')


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    model_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    cfg = json.load(open(os.path.join(model_dir, 'config.json'), encoding='utf-8'))
    token_map, _ = load_vocab(model_dir)
    tensors = read_safetensors(os.path.join(model_dir, 'model.safetensors'))

    tokens = ['<pad>', '<unk>', '<eos>']
    for tok in token_map.keys():
        text = token_map[tok]
        if text is not None and len(text) == 1 and text not in ('<', '>', '|'):
            tokens.append(text)
    vocab_size = len(tokens)
    char_to_new = {tok: i for i, tok in enumerate(tokens) if i >= 3}
    print(f'新词表 {vocab_size}（单字 {vocab_size - 3}）')

    wte = tensors['transformer.wte.weight']
    new_wte = np.zeros((vocab_size, wte.shape[1]), dtype=wte.dtype)
    for tok, old_id in token_map.items():
        text = token_map[tok] if isinstance(token_map, dict) else None
        new_id = char_to_new.get(text) if text else None
        if new_id is not None and new_id >= 3:
            new_wte[new_id] = wte[old_id]
    eot = token_map.get('<|endoftext|>', 0)
    new_wte[0] = wte[min(eot, wte.shape[0] - 1)]
    new_wte[1] = wte.mean(axis=0)
    new_wte[2] = wte[min(eot, wte.shape[0] - 1)]

    out = {
        'tok_emb.weight': new_wte.astype(np.float16),
        'pos_emb.weight': tensors['transformer.wpe.weight'].astype(np.float16),
        'ln_f.weight': tensors['transformer.ln_f.weight'].astype(np.float16),
        'ln_f.bias': tensors['transformer.ln_f.bias'].astype(np.float16),
    }
    if 'lm_head.weight' in tensors and not np.array_equal(
        tensors['lm_head.weight'], tensors['transformer.wte.weight']
    ):
        # 输出头不与输入嵌入共享：按新词表重映射成 head.weight（共享时不写，加载器回退 tok_emb）
        lm = tensors['lm_head.weight']
        new_head = np.zeros((vocab_size, lm.shape[1]), dtype=lm.dtype)
        for tok, old_id in token_map.items():
            text = token_map[tok] if isinstance(token_map, dict) else None
            new_id = char_to_new.get(text) if text else None
            if new_id is not None and new_id >= 3:
                new_head[new_id] = lm[old_id]
        out['head.weight'] = new_head.astype(np.float16)
        print('检测到独立输出头，已重映射为 head.weight')
    for i in range(cfg['n_layer']):
        p, q = f'transformer.h.{i}.', f'blocks.{i}.'
        for src, dst, tr in [
            ('ln_1.weight', 'ln1.weight', False), ('ln_1.bias', 'ln1.bias', False),
            ('attn.c_attn.weight', 'attn.qkv.weight', True), ('attn.c_attn.bias', 'attn.qkv.bias', False),
            ('attn.c_proj.weight', 'attn.proj.weight', True), ('attn.c_proj.bias', 'attn.proj.bias', False),
            ('ln_2.weight', 'ln2.weight', False), ('ln_2.bias', 'ln2.bias', False),
            ('mlp.c_fc.weight', 'mlp.fc.weight', True), ('mlp.c_fc.bias', 'mlp.fc.bias', False),
            ('mlp.c_proj.weight', 'mlp.proj.weight', True), ('mlp.c_proj.bias', 'mlp.proj.bias', False),
        ]:
            arr = tensors[p + src]
            out[q + dst] = np.ascontiguousarray((arr.T if tr else arr).astype(np.float16))

    blob = bytearray()
    hdr = {}
    offset = 0
    for name, arr in out.items():
        raw = arr.tobytes()
        while offset % 8 != 0:
            blob.append(0)
            offset += 1
        hdr[name] = {'dtype': 'F16', 'shape': list(arr.shape), 'data_offsets': [offset, offset + len(raw)]}
        blob.extend(raw)
        offset += len(raw)
    hj = json.dumps(hdr, separators=(',', ':')).encode('utf-8')
    while len(hj) % 8 != 0:
        hj += b' '
    with open(os.path.join(out_dir, 'model.safetensors'), 'wb') as f:
        f.write(struct.pack('<Q', len(hj)))
        f.write(hj)
        f.write(bytes(blob))
    json.dump(
        {
            'vocab_size': vocab_size,
            'n_layer': cfg['n_layer'],
            'n_embd': cfg['n_embd'],
            'n_head': cfg['n_head'],
            'context': cfg['n_positions'],
        },
        open(os.path.join(out_dir, 'config.json'), 'w'),
        ensure_ascii=False,
    )
    json.dump({'tokens': tokens}, open(os.path.join(out_dir, 'vocab.json'), 'w'), ensure_ascii=False)
    print(f'三件套已写出: {out_dir}/{{config.json, vocab.json, model.safetensors}}')


if __name__ == '__main__':
    main()
