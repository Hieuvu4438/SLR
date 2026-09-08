#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
target="${repo_root}/third_party/SEDS"
revision="434e3f714fcb6a7d1f4001fb9a246bbd93ec0246"
remote="https://github.com/longtaojiang/SEDS.git"

if [[ -e "${target}" && ! -d "${target}/.git" ]]; then
  echo "Refusing to replace non-git path: ${target}" >&2
  exit 2
fi

if [[ ! -d "${target}/.git" ]]; then
  git clone --filter=blob:none "${remote}" "${target}"
fi

if [[ -n "$(git -C "${target}" status --porcelain)" ]]; then
  echo "Refusing to change dirty SEDS checkout: ${target}" >&2
  exit 2
fi

git -C "${target}" fetch --depth 1 origin "${revision}"
git -C "${target}" checkout --detach "${revision}"
actual="$(git -C "${target}" rev-parse HEAD)"
if [[ "${actual}" != "${revision}" ]]; then
  echo "SEDS revision mismatch: expected ${revision}, got ${actual}" >&2
  exit 2
fi

echo "SEDS ready at ${target} (${actual})"
