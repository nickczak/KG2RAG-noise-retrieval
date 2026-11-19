#!/usr/bin/env bash
# Run from any directory; pass --resume to continue the same experiment.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
# Read launch settings from the adjacent JSON so there is one source of truth.
exec .venv/bin/python -u - "$@" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

settings = json.loads(Path('output/sem50_t03/run_settings.json').read_text())
if sys.argv[1:] not in ([], ['--resume']):
    raise SystemExit('Usage: bash output/sem50_t03/run.sh [--resume]')
command = [sys.executable, '-u', 'code/kg_rag_distractor.py']
for key, value in settings.items():
    command.extend(['--' + key, str(value)])
command.extend(sys.argv[1:])
raise SystemExit(subprocess.call(command))
PY
