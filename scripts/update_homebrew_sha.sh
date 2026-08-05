#!/usr/bin/env bash
# Refresh the sha256 in homebrew/ai-rganize.rb for a given release tag.
# Usage: ./scripts/update_homebrew_sha.sh v1.0.0
set -euo pipefail

TAG="${1:?Usage: $0 v1.0.0}"
FORMULA="$(cd "$(dirname "$0")/.." && pwd)/homebrew/ai-rganize.rb"
URL="https://github.com/adefemi171/airganizer/archive/refs/tags/${TAG}.tar.gz"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "Downloading ${URL}"
curl -fsSL "$URL" -o "${tmpdir}/src.tar.gz"
SHA="$(shasum -a 256 "${tmpdir}/src.tar.gz" | awk '{print $1}')"
echo "sha256: ${SHA}"

if [[ "$(uname)" == "Darwin" ]]; then
  sed -i '' -E "s/sha256 \"[0-9a-fA-F]{64}\"/sha256 \"${SHA}\"/" "$FORMULA"
  sed -i '' -E "s|archive/refs/tags/v[^\"]+\.tar\.gz|archive/refs/tags/${TAG}.tar.gz|" "$FORMULA"
else
  sed -i -E "s/sha256 \"[0-9a-fA-F]{64}\"/sha256 \"${SHA}\"/" "$FORMULA"
  sed -i -E "s|archive/refs/tags/v[^\"]+\.tar\.gz|archive/refs/tags/${TAG}.tar.gz|" "$FORMULA"
fi

echo "Updated ${FORMULA}"
echo "Next: copy to your homebrew-airganize tap and commit."
