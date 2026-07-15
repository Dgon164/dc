#!/bin/bash
set -euo pipefail

# Only needed in Claude Code on the web — local checkouts manage their own env.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Idempotent: pip skips requirements that are already satisfied.
python3 -m pip install --quiet --root-user-action=ignore -r requirements.txt

# Battery output uses unicode glyphs; force UTF-8 stdio everywhere
# (README calls this out for Windows; harmless on Linux).
echo 'export PYTHONUTF8=1' >> "$CLAUDE_ENV_FILE"
