#!/usr/bin/env bash
# Assemble the static site in web/www: wasm, ONNX model and example X-rays.
#   web/build.sh path/to/efficientnetv2-b0.onnx      (the .json from xray-export sits next to it)
# Then serve web/www with any static server, e.g. `python -m http.server -d web/www`.
set -euo pipefail
onnx=$(realpath "${1:?usage: web/build.sh path/to/model.onnx}")
cd "$(dirname "$0")"

# Build scripts need a host linker; on Windows without MSVC use the GNU toolchain.
toolchain=()
if [[ "${OS:-}" == Windows_NT ]] && rustup toolchain list | grep -q windows-gnu; then
  toolchain=(+stable-x86_64-pc-windows-gnu)
fi
cargo "${toolchain[@]}" build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/xray_wasm.wasm www/

mkdir -p www/model www/examples
cp "$onnx" www/model/model.onnx
cp "${onnx%.onnx}.json" www/model/model.json
i=0; for f in ../space/examples/normal_*.jpeg; do i=$((i+1)); cp "$f" "www/examples/normal_$i.jpeg"; done
i=0; for f in ../space/examples/pneumonia_*.jpeg; do i=$((i+1)); cp "$f" "www/examples/pneumonia_$i.jpeg"; done
du -sh www/xray_wasm.wasm www/model/model.onnx
