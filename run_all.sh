#!/bin/bash

# ==========================================
# Agro-Suggester — Start All Services
# Each service opens in its own terminal
# ==========================================

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Terminal launcher ──────────────────────────────────────────
# Tries: iTerm2 → Terminal.app → gnome-terminal → xterm
open_terminal() {
  local title="$1"
  local cmd="$2"

  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: prefer iTerm2, fall back to Terminal.app
    if open -Ra "iTerm" 2>/dev/null; then
      osascript <<EOF
tell application "iTerm"
  create window with default profile
  tell current session of current window
    write text "printf '\\\\033]0;${title}\\\\007'; ${cmd}"
  end tell
end tell
EOF
    else
      osascript <<EOF
tell application "Terminal"
  do script "printf '\\\\033]0;${title}\\\\007'; ${cmd}"
  set custom title of front window to "${title}"
end tell
EOF
    fi

  else
    # Linux: try common terminal emulators in order
    if command -v gnome-terminal &>/dev/null; then
      gnome-terminal --title="$title" -- bash -c "$cmd; exec bash"
    elif command -v konsole &>/dev/null; then
      konsole --new-tab --title "$title" -e bash -c "$cmd; exec bash"
    elif command -v xfce4-terminal &>/dev/null; then
      xfce4-terminal --title="$title" -e "bash -c '$cmd; exec bash'"
    elif command -v xterm &>/dev/null; then
      xterm -title "$title" -e bash -c "$cmd; exec bash" &
    else
      echo "No supported terminal emulator found. Install gnome-terminal or xterm."
      exit 1
    fi
  fi

  sleep 0.4   # small gap so windows don't spawn on top of each other
}

# ── Shared venv setup (runs inline, not in a new window) ───────
echo "=========================================="
echo "  Agro-Suggester — Starting Services"
echo "=========================================="
echo ""
echo "[0/4] Setting up virtual environment..."
cd "$ROOT_DIR"
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi
source venv/bin/activate
echo "      Done."
echo ""

# ── Service 1 — Recommendation (port 8002) ────────────────────
echo "[1/4] Opening Recommendation Service terminal..."
open_terminal "Recommendation :8002" \
  "cd '$ROOT_DIR' && source venv/bin/activate && cd recommendation && pip install -r requirements.txt -q && echo '' && echo '  Recommendation Service — http://localhost:8002' && echo '' && python -m uvicorn app.main:app --host 0.0.0.0 --port 8002"

# ── Service 2 — Input Prep (port 8001) ────────────────────────
echo "[2/4] Opening Input Prep Service terminal..."
open_terminal "InputPrep :8001" \
  "cd '$ROOT_DIR' && source venv/bin/activate && cd input_prep && pip install -r requirements.txt -q && echo '' && echo '  Input Prep Service — http://localhost:8001' && echo '' && python -m uvicorn app.main:app --host 0.0.0.0 --port 8001"

# ── Service 3 — API Gateway (port 8000) ───────────────────────
echo "[3/4] Opening API Gateway terminal..."
open_terminal "Gateway :8000" \
  "cd '$ROOT_DIR' && source venv/bin/activate && cd gateway && pip install -r requirements.txt -q && echo '' && echo '  API Gateway — http://localhost:8000' && echo '' && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

# ── Service 4 — React Frontend (port 5173) ────────────────────
echo "[4/4] Opening Frontend terminal..."
open_terminal "Frontend :5173" \
  "cd '$ROOT_DIR/frontend' && npm install --silent && echo '' && echo '  Frontend — http://localhost:5173' && echo '' && VITE_API_GATEWAY=http://localhost:8000 npm run dev"

echo ""
echo "=========================================="
echo "  All terminals launched!"
echo ""
echo "  Recommendation  → http://localhost:8002"
echo "  Input Prep      → http://localhost:8001"
echo "  API Gateway     → http://localhost:8000"
echo "  Frontend        → http://localhost:5173"
echo "=========================================="