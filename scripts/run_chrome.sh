#!/usr/bin/env bash
set -e

PORT=${1:-9222}

# Use a dedicated profile directory for remote debugging.
# Chrome 94+ silently refuses --remote-debugging-port when using the real
# user profile for security reasons.  A separate dir launches its own
# instance alongside your normal browser, fully independent.
DEBUG_PROFILE_DIR="${HOME}/.config/chrome-debug-profile"

echo "======================================================="
echo "Launching Google Chrome with Remote Debugging on :$PORT"
echo "Debug profile: $DEBUG_PROFILE_DIR"
echo "NOTE: Log in to Claude, ChatGPT, Gemini, and DeepSeek"
echo "      in the window that opens — sessions are saved."
echo "======================================================="

# Kill any Chrome process using the debug profile or the CDP port
echo "Cleaning up any existing Chrome debug session..."
pkill -f "chrome-debug-profile" 2>/dev/null || true
pkill -f -- "--remote-debugging-port=${PORT}" 2>/dev/null || true

# Wait until the port is actually free
for i in $(seq 1 20); do
    if ! ss -tlnp | grep -q ":${PORT} "; then
        break
    fi
    sleep 0.5
done

# Detect Chrome binary
if command -v google-chrome &> /dev/null; then
    CHROME_BIN="google-chrome"
elif command -v google-chrome-stable &> /dev/null; then
    CHROME_BIN="google-chrome-stable"
elif command -v chromium-browser &> /dev/null; then
    CHROME_BIN="chromium-browser"
elif command -v chromium &> /dev/null; then
    CHROME_BIN="chromium"
else
    echo "Error: Google Chrome or Chromium not found in PATH."
    exit 1
fi

mkdir -p "$DEBUG_PROFILE_DIR"

"$CHROME_BIN" \
  --remote-debugging-port="$PORT" \
  --remote-debugging-address=127.0.0.1 \
  --user-data-dir="$DEBUG_PROFILE_DIR" \
  --no-first-run \
  --no-default-browser-check \
  > /dev/null 2>&1 &

CHROME_PID=$!
echo "Chrome started (PID: $CHROME_PID). Waiting for CDP on port $PORT..."

for i in $(seq 1 30); do
    sleep 0.5
    if curl -s --max-time 1 "http://127.0.0.1:${PORT}/json/version" > /dev/null 2>&1; then
        echo "✓ CDP is ready on port $PORT!"
        echo ""
        echo "Next steps:"
        echo "  1. Log in to Claude, ChatGPT, Gemini, DeepSeek in the Chrome window"
        echo "  2. Then run: uv run python -m debate check-cdp"
        exit 0
    fi
done

echo "⚠  CDP port $PORT is not responding after 15 seconds."
echo "Try running manually: google-chrome --remote-debugging-port=$PORT --user-data-dir=$DEBUG_PROFILE_DIR"

