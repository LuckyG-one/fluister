#!/usr/bin/env bash
# Fluister installer: creates .venv, installs dependencies, creates your
# config.yaml and (optionally) starts Fluister at login.
#
#   ./install.sh              install
#   ./install.sh --autostart  install and launch at login (macOS)
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"

PYTHON="${PYTHON:-python3}"
LLM_MODEL="qwen3:4b-instruct"
LABEL="io.github.luckyg-one.fluister"

say() { printf '\033[1m==> %s\033[0m\n' "$1"; }

if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
  echo "Fluister needs Python 3.10 or newer. Set PYTHON=/path/to/python3.11 and retry." >&2
  exit 1
fi

say "Creating virtual environment"
"$PYTHON" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip

say "Installing dependencies"
.venv/bin/pip install --quiet -r requirements.txt

APPLE_SILICON=0
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  APPLE_SILICON=1
  say "Apple Silicon detected: installing parakeet-mlx (streaming ASR)"
  .venv/bin/pip install --quiet parakeet-mlx mlx-whisper
fi

if [ ! -f config.yaml ]; then
  say "Creating config.yaml from config.example.yaml"
  cp config.example.yaml config.yaml
  if [ "$APPLE_SILICON" = "0" ]; then
    # parakeet-mlx is Apple Silicon only; fall back to faster-whisper.
    .venv/bin/python - <<'PY'
from pathlib import Path
path = Path("config.yaml")
path.write_text(path.read_text().replace("backend: parakeet", "backend: fasterwhisper")
                .replace("model: mlx-community/parakeet-tdt-0.6b-v3", "model: large-v3-turbo"))
PY
  fi
fi

if command -v ollama >/dev/null 2>&1; then
  say "Pulling the local formatting model ($LLM_MODEL)"
  ollama pull "$LLM_MODEL"
else
  echo "Ollama not found. Install it from https://ollama.com and run: ollama pull $LLM_MODEL"
  echo "(or point the llm block in config.yaml at any OpenAI-compatible API)"
fi

if [ "${1:-}" = "--autostart" ]; then
  if [ "$(uname -s)" != "Darwin" ]; then
    echo "--autostart is macOS only." >&2
    exit 1
  fi
  say "Installing launch agent ($LABEL)"
  PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
  LOG_DIR="$HOME/Library/Logs/Fluister"
  mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents"
  cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$ROOT/.venv/bin/python</string>
        <string>-m</string>
        <string>fluister.main</string>
    </array>
    <key>WorkingDirectory</key><string>$ROOT</string>
    <key>RunAtLoad</key><true/>
    <!-- Restart after a crash only, so "Quit Fluister" in the menu really quits. -->
    <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
    <key>LimitLoadToSessionType</key><string>Aqua</string>
    <key>StandardOutPath</key><string>$LOG_DIR/fluister.log</string>
    <key>StandardErrorPath</key><string>$LOG_DIR/fluister.log</string>
</dict>
</plist>
PLIST
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$PLIST"
  echo "Fluister now starts at login. Logs: $LOG_DIR/fluister.log"
fi

say "Done"
echo "Start it with:  .venv/bin/python -m fluister.main"
echo "Then hold right Option, speak, release. macOS asks for Microphone,"
echo "Accessibility and Input Monitoring permission on first use."
