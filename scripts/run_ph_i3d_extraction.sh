#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo_root/methods/elsc/scripts/run_ph_i3d_extraction.sh" "$@"
