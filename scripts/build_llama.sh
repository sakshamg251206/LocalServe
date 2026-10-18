#!/usr/bin/env bash
# Build llama.cpp with Metal at the commit pinned in configs/llama_cpp.yaml, into .vendor/llama.cpp.
set -euo pipefail
cd "$(dirname "$0")/.."
cfg=configs/llama_cpp.yaml
repo=$(awk '/^repo:/{print $2}' "$cfg")
commit=$(awk '/^commit:/{print $2}' "$cfg")
dir=.vendor/llama.cpp
[ -d "$dir/.git" ] || git clone --quiet "$repo" "$dir"
git -C "$dir" fetch --quiet origin "$commit" 2>/dev/null || true
git -C "$dir" checkout --quiet "$commit"
cmake -S "$dir" -B "$dir/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build "$dir/build" --target llama-server llama-bench llama-perplexity llama-cli llama-quantize -j "$(sysctl -n hw.ncpu)"
"$dir/build/bin/llama-server" --version
