#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT}/data/replica_v1"
ARCHIVE="${DATA_DIR}/office_0.zip"

command -v gdown >/dev/null 2>&1 || {
  echo "gdown is required. Install it with: python -m pip install gdown" >&2
  exit 1
}
command -v unzip >/dev/null 2>&1 || {
  echo "unzip is required." >&2
  exit 1
}

mkdir -p "${DATA_DIR}"
python -m gdown \
  "https://drive.google.com/uc?id=1MZzwlHga5B29KMvj6uMNR9QfNvoomENY" \
  --output "${ARCHIVE}"
unzip -q -o "${ARCHIVE}" -d "${DATA_DIR}"
rm -f "${ARCHIVE}"

echo "Replica office0 is available under ${DATA_DIR}/office_0"
