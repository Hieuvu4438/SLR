#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${repo_root}/third_party/SLRT"
commit="38a4f7b00da7a858d59b7fabe5093876a84db8e0"

if [[ -d "${target}/.git" ]]; then
  actual="$(git -C "${target}" rev-parse HEAD)"
  if [[ "${actual}" != "${commit}" ]]; then
    echo "Existing SLRT checkout is at ${actual}, expected ${commit}" >&2
    exit 2
  fi
  exit 0
fi

mkdir -p "${repo_root}/third_party"
git clone --filter=blob:none --no-checkout https://github.com/FangyunWei/SLRT.git "${target}"
git -C "${target}" sparse-checkout init --cone
git -C "${target}" sparse-checkout set CiCo
git -C "${target}" checkout --detach "${commit}"
test "$(git -C "${target}" rev-parse HEAD)" = "${commit}"
