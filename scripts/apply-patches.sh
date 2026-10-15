#!/usr/bin/env bash
# Apply the Drama 3 GB10 patches, with a backup of every file touched.
# Every patch is default OFF, so applying them changes nothing until a flag is set.
#
#   scripts/apply-patches.sh [site-packages-root]
#
# With no argument the root is derived from the importable sglang_omni.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${1:-$(python -c 'import os,sglang_omni;print(os.path.dirname(os.path.dirname(sglang_omni.__file__)))')}"
model="$root/sglang_omni/models/fishaudio_drama3"
[ -d "$model" ] || { echo "no Drama 3 model dir under $root" >&2; exit 1; }

stamp="$(date +%Y%m%d-%H%M%S)"
for f in sampling_kernels.py tower_kernels.py quantization.py sglang_model.py; do
  cp -n "$model/$f" "$model/$f.pre-gb10-recipe-$stamp"
done
echo "backups: $model/*.pre-gb10-recipe-$stamp"

# Dry run everything before touching anything. 0004 is NOT dry-run here: it
# builds on 0002's version of quantization.py, so a dry run against the pristine
# tree would fail by design. It is applied below, after 0002.
for p in 0001-sm121-kernel-gate-opt-in 0002-output-head-online-fp8 \
         0003-audio-decoder-mtp-online-fp8; do
  patch -d "$root" --dry-run --fuzz=0 -p1 < "$here/patches/$p.patch"
done

for p in 0001-sm121-kernel-gate-opt-in 0002-output-head-online-fp8 0004-backbone-online-fp8; do
  patch -d "$root" --fuzz=0 -p1 < "$here/patches/$p.patch"
done
cp "$here/patches/0003-mtp_fp8.py.new" "$model/mtp_fp8.py"
patch -d "$root" --fuzz=0 -p1 < "$here/patches/0003-audio-decoder-mtp-online-fp8.patch"

python -c 'import compileall,sys; sys.exit(0 if compileall.compile_dir(sys.argv[1], quiet=2) else 1)' "$model"
echo "applied. all flags still default OFF; see README for the launch env."
