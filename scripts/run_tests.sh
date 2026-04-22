# scripts/run_tests.sh
#!/usr/bin/env bash
# Run pytest in the project's venv. Reads venv path from .venv-path or env.
set -euo pipefail

VENV="${PROJECT_VENV:-}"
if [[ -z "${VENV}" && -f .venv-path ]]; then
    VENV="$(cat .venv-path)"
fi
VENV="${VENV:-$HOME/.venvs/rapids25.10_python3.12_cuda13}"

PY="${VENV}/bin/python"
if [[ ! -x "${PY}" ]]; then
    echo "No interpreter at ${PY}" >&2
    exit 2
fi

TAIL="${TEST_TAIL:-50}"

set -o pipefail
"${PY}" -m pytest "$@" 2>&1 | tail -n "${TAIL}"