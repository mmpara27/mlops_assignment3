#!/bin/sh
set -eu
TASK_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$TASK_DIR"
exec "$TASK_DIR/.venv/bin/python" -u "$TASK_DIR/h2o_automl.py" --data "$TASK_DIR/athletes.csv" "$@"
