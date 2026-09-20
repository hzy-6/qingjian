#!/usr/bin/env bash
# 把本机 data/generated/ 发成一版不可变的数据 Release（data-vN，预发布），并写 tools/release/data.lock。
# CI 与自编译按锁文件取数据（data-fetch.sh）；改了数据发新号，锁文件与用到新数据的代码同一个提交。
#
#   tools/release/data-bundle.sh                 # 发到下一个 data-vN
#   tools/release/data-bundle.sh --tag data-v7   # 指定标签；已存在就拒绝
#   tools/release/data-bundle.sh --pack          # 只打包到 target/release-data/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/target/release-data"
LOCK="$ROOT/tools/release/data.lock"
cd "$ROOT"

MODE=upload
TAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pack) MODE=pack; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    *) echo "未知参数 $1" >&2; exit 1 ;;
  esac
done

PRODUCT_FILES=(dict.qj lm.qj glossary-en.qj glossary-ja.qj glossary-zh.qj glossary-es.qj english.tsv english-frequency.tsv)
LLM_FILES=(gloss-llm.jsonl gloss-en-llm.jsonl pinyin-llm.jsonl)
# 随包整句模型:挑裁剪过的 *-text.gguf(与 bundle.sh 同规则);没有就发失败——CI 发的包必须带模型
MODEL_GGUF=""
for g in data/model/*-text.gguf data/model/*.gguf; do
  [[ -f "$g" ]] || continue
  MODEL_GGUF="$g"; break
done
[[ -n "$MODEL_GGUF" ]] || { echo "缺少 data/model/*.gguf(整句模型,裁剪版优先),CI 发的包必须带模型" >&2; exit 1; }

for f in "${PRODUCT_FILES[@]}"; do
  [[ -f "data/generated/$f" ]] || { echo "缺少 data/generated/$f，先按 assets/lexicon/QINGJIAN.md 生成" >&2; exit 1; }
done
DOMAIN_FILES=()
for f in data/generated/dicts/*.qj; do [[ -f "$f" ]] && DOMAIN_FILES+=("dicts/$(basename "$f")"); done
[[ ${#DOMAIN_FILES[@]} -gt 0 ]] || { echo "缺少 data/generated/dicts/*.qj" >&2; exit 1; }

rm -rf "$OUT" && mkdir -p "$OUT"
tar -czf "$OUT/qingjian-data.tar.gz" -C data/generated "${PRODUCT_FILES[@]}" "${DOMAIN_FILES[@]}"
present=()
for f in "${LLM_FILES[@]}"; do [[ -f "data/generated/$f" ]] && present+=("$f"); done
[[ ${#present[@]} -gt 0 ]] && tar -czf "$OUT/qingjian-llm-intermediates.tar.gz" -C data/generated "${present[@]}"
cp "$MODEL_GGUF" "$OUT/"
(cd "$OUT" && shasum -a 256 ./*.tar.gz ./*.gguf | tee SHA256SUMS)
du -h "$OUT"/*.tar.gz "$OUT"/*.gguf

[[ "$MODE" == "pack" ]] && exit 0

if [[ -z "$TAG" ]]; then
  last="$(gh release list --limit 200 --json tagName --jq '.[].tagName' | grep -E '^data-v[0-9]+$' | sed 's/data-v//' | sort -n | tail -1)"
  TAG="data-v$(( ${last:-0} + 1 ))"
fi
[[ "$TAG" =~ ^data-v[0-9]+$ ]] || { echo "标签要写成 data-vN：$TAG" >&2; exit 1; }
gh release view "$TAG" >/dev/null 2>&1 && { echo "$TAG 已存在，数据版本不覆盖" >&2; exit 1; }

sha_of() { grep " ./$1\$" "$OUT/SHA256SUMS" | cut -d' ' -f1; }
gh release create "$TAG" --prerelease --target "$(git rev-parse HEAD)" --title "产品数据 $TAG" \
  --notes "词库 / 语言模型 / 释义表（qingjian-data.tar.gz）、整句模型 GGUF、LLM 续跑中间产物（qingjian-llm-intermediates.tar.gz）。不可变；仓库 tools/release/data.lock 钉住要用哪一版。" \
  "$OUT"/*.tar.gz "$OUT"/*.gguf "$OUT/SHA256SUMS"

cat > "$LOCK" <<EOF
# 产品数据版本，data-bundle.sh 写、data-fetch.sh 读；不要手改
tag = $TAG
qingjian-data.tar.gz = $(sha_of qingjian-data.tar.gz)
$(basename "$MODEL_GGUF") = $(sha_of "$(basename "$MODEL_GGUF")")
EOF
echo "已发 $TAG，锁文件已更新（记得提交）"
