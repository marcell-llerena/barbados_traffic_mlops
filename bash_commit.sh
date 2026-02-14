#!/usr/bin/env bash
# Commit only updated files. Run from project root: ./bash_commit.sh
set -e

cd "$(dirname "$0")"

git add pyproject.toml
git commit -m "chore: update project configuration and dependencies"

git add uv.lock
git commit -m "chore: update dependency lock file for reproducible builds"

git add src/traffic_mlops/ingestion/tracker.py
git commit -m "feat: add YOLO vehicle tracker"

git add bash_commit.sh
git commit -m "chore: add script for incremental file commits"
