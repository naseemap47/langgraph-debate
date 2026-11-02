#!/usr/bin/env bash
set -e

PORT=${1:-9222}
USER_DATA_DIR="${HOME}/.config/google-chrome"

echo "======================================================="
echo "Launching Google Chrome with Remote Debugging on :$PORT"
echo "Profile directory: $USER_DATA_DIR"
echo "======================================================="

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

"$CHROME_BIN" --remote-debugging-port="$PORT" --user-data-dir="$USER_DATA_DIR" > /dev/null 2>&1 &
echo "Chrome started in background (PID: $!)."
echo "You can now run 'python -m debate check-cdp' to verify connection."
