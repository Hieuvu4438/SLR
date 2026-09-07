#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo_root/methods/elsc/scripts/run_csl_multiseed.sh" "$@"
