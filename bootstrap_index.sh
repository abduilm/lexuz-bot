#!/usr/bin/env bash
set -euo pipefail

# Where we keep the index on the Render Disk
INDEX_DIR="${INDEX_DIR:-/var/data/index_store2}"
ZIP_URL="${INDEX_URL:?INDEX_URL env var must be set}"
ZIP_PATH="/var/data/index_store2.zip"

echo "INDEX_DIR=$INDEX_DIR"
mkdir -p /var/data

# If we already have embeddings, skip download
if [ -f "$INDEX_DIR/embeddings.npy" ] && [ -f "$INDEX_DIR/meta.jsonl" ]; then
  echo "Index already present at $INDEX_DIR — skipping download."
else
  echo "Downloading index zip from Google Drive..."
  # Install gdown (handles Google Drive downloads)
  pip install --no-cache-dir gdown >/dev/null

  # Clean any partial previous files
  rm -f "$ZIP_PATH"
  rm -rf "$INDEX_DIR"

  # Download
  gdown --fuzzy "$ZIP_URL" -O "$ZIP_PATH"

  echo "Unzipping to $INDEX_DIR..."
  mkdir -p "$INDEX_DIR"
  unzip -o "$ZIP_PATH" -d /var/data/ >/dev/null

  # If the zip contains index_store2/ at top-level, ensure INDEX_DIR points there
  # (Most zips will unpack to /var/data/index_store2/ directly)
  echo "Unzip complete."
fi

echo "Starting app..."
# Ensure your app reads INDEX_DIR from env (your app.py already does)
exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
