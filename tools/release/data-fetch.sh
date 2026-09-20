#!/usr/bin/env bash
# 按 tools/release/data.lock 下载产品数据并校验:qingjian-data.tar.gz 解到 data/generated/,GGUF 模型放 data/model/。
#
#   tools/release/data-fetch.sh            # 下载 + 校验 + 解开
#   tools/release/data-fetch.sh --verify   # 只校验 target/release-data/ 里已下载的文件
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOCK="$ROOT/tools/release/data.lock"
OUT="$ROOT/target/release-data"
REPO="${QINGJIAN_DATA_REPO:-hzy-6/qingjian}"
ASSETS=(qingjian-data.tar.gz)
cd "$ROOT"

[[ -f "$LOCK" ]] || { echo "缺少 $LOCK" >&2; exit 1; }
lock_value() { grep -E "^$1 *= *" "$LOCK" | head -1 | sed -E 's/^[^=]*= *//' | tr -d '[:space:]'; }
sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }

TAG="$(lock_value tag)"
[[ -n "$TAG" ]] || { echo "$LOCK 里没有 tag" >&2; exit 1; }
mkdir -p "$OUT"
# 模型资产:lock 里的 gguf 条目(键名含 .gguf);旧版 lock 没有模型条目时报警退出——发版包必须带模型
MODEL_ASSET="$(grep -E '^[^#=]+\.gguf *= ' "$LOCK" | head -1 | sed -E 's/^([^=]+) *=.*/\1/' | tr -d '[:space:]')"
[[ -n "$MODEL_ASSET" ]] || { echo "$LOCK 里没有模型(.gguf)条目,旧版数据包不带模型;发新数据版带上模型再构建" >&2; exit 1; }
ASSETS=(qingjian-data.tar.gz "$MODEL_ASSET")

if [[ "${1:-}" != "--verify" ]]; then
  for f in "${ASSETS[@]}"; do
    rm -f "$OUT/$f"
    if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
      gh release download "$TAG" --repo "$REPO" --pattern "$f" --dir "$OUT"
    else
      curl -fL --retry 3 -o "$OUT/$f" "https://github.com/$REPO/releases/download/$TAG/$f"
    fi
  done
fi

for f in "${ASSETS[@]}"; do
  expected="$(lock_value "$f")"
  actual="$(sha256 "$OUT/$f")"
  [[ -n "$expected" && "$actual" == "$expected" ]] || { echo "$f 与 data.lock 不符（$TAG）：期望 $expected，实际 $actual" >&2; exit 1; }
  echo "$f  $actual"
done
[[ "${1:-}" == "--verify" ]] && exit 0

mkdir -p data/generated data/model
tar -xzf "$OUT/qingjian-data.tar.gz" -C data/generated
cp "$OUT/$MODEL_ASSET" "data/model/$MODEL_ASSET"
# 解出来的 mtime 比 checkout 出来的 TSV 旧，bundle.sh 会以为要重打
find data/generated data/model -type f -exec touch {} +
echo "产品数据 $TAG 已就位"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "DATA_TAG=$TAG"
    echo "DATA_SHA256=$(lock_value qingjian-data.tar.gz)"
    echo "MODEL_SHA256=$(lock_value "$MODEL_ASSET")"
  } >> "$GITHUB_ENV"
fi
