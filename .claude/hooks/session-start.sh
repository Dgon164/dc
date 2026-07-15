#!/bin/bash
set -euo pipefail

# Only needed in Claude Code on the web — local checkouts manage their own env.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Idempotent: pip skips requirements that are already satisfied.
python3 -m pip install --quiet --root-user-action=ignore -r requirements.txt

# test_intake.py uses 3.12+ f-string syntax (PEP 701); the container's
# default python3 is 3.11. Keep 3.13 provisioned so that battery can run
# as: python3.13 test_intake.py
if command -v python3.13 >/dev/null 2>&1; then
  python3.13 -m pip install --quiet --root-user-action=ignore -r requirements.txt
fi

{
  # Battery output uses unicode glyphs; force UTF-8 stdio everywhere
  # (README calls this out for Windows; harmless on Linux).
  echo 'export PYTHONUTF8=1'
  # Isaac's multicore model is process-level parallelism (workers on
  # their own cores). On this 4-core container, BLAS threads inside each
  # worker's numpy only oversubscribe the cores and slow the fleet ~3x
  # (test_multicore M4b: 0.31x -> 3.39x with pinning). One thread each.
  echo 'export OPENBLAS_NUM_THREADS=1'
  echo 'export OMP_NUM_THREADS=1'
  echo 'export MKL_NUM_THREADS=1'
} >> "$CLAUDE_ENV_FILE"
