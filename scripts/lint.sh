#!/bin/bash
set -e

if [ "$1" == "--check" ]; then
    black . --exclude=notebooks --exclude=.venv --check
    ruff check --target-version=py310 .
else
    black . --exclude=notebooks --exclude=.venv
    ruff check --target-version=py310 --fix .
fi
