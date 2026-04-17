#!/usr/bin/env bash
# playtest.sh — Visual playtesting tool for Pac-Man 3D
#
# Usage:
#   ./playtest.sh start              Launch game, print window ID
#   ./playtest.sh screenshot [name]  Capture screenshot, print path
#   ./playtest.sh key <Key>          Send a single keypress (e.g. Left, Right, Up, Down, space)
#   ./playtest.sh keys "K1 K2 ..."   Send multiple keys with 100ms gaps
#   ./playtest.sh play <secs>        Send random inputs for N seconds, screenshot at end
#   ./playtest.sh stop               Kill the game
#   ./playtest.sh status             Check if game is running
#
# Screenshots are saved to /tmp/pac-playtest/ and can be viewed with Claude's Read tool.

set -euo pipefail

GAME_DIR="$(cd "$(dirname "$0")" && pwd)"
GAME_BIN="$GAME_DIR/target/release/pac"
PID_FILE="/tmp/pac-playtest.pid"
WINDOW_FILE="/tmp/pac-playtest.wid"
SCREENSHOT_DIR="/tmp/pac-playtest"
DISPLAY="${DISPLAY:-:1}"
export DISPLAY

mkdir -p "$SCREENSHOT_DIR"

die() { echo "ERROR: $*" >&2; exit 1; }

find_window() {
    # Try to find the Pac-Man window, retry a few times
    for _ in 1 2 3 4 5; do
        local wid
        wid=$(xdotool search --name "Pac-Man 3D" 2>/dev/null | tail -1) || true
        if [[ -n "$wid" ]]; then
            echo "$wid"
            return 0
        fi
        sleep 0.5
    done
    return 1
}

cmd_start() {
    # Kill existing instance if any
    if [[ -f "$PID_FILE" ]]; then
        local old_pid
        old_pid=$(cat "$PID_FILE")
        kill "$old_pid" 2>/dev/null || true
        rm -f "$PID_FILE" "$WINDOW_FILE"
        sleep 0.5
    fi

    # Build if needed
    if [[ ! -f "$GAME_BIN" ]] || [[ "$GAME_DIR/src/main.rs" -nt "$GAME_BIN" ]]; then
        echo "Building game..."
        (cd "$GAME_DIR" && cargo build --release 2>&1) || die "Build failed"
    fi

    # Launch
    "$GAME_BIN" &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    sleep 1

    # Find window
    local wid
    wid=$(find_window) || die "Could not find game window"
    echo "$wid" > "$WINDOW_FILE"

    echo "Game started: PID=$pid WID=$wid"
    echo "Take screenshots with: ./playtest.sh screenshot"
}

cmd_stop() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        kill "$pid" 2>/dev/null && echo "Game stopped (PID=$pid)" || echo "Game was not running"
        rm -f "$PID_FILE" "$WINDOW_FILE"
    else
        echo "No game running"
    fi
}

cmd_status() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Game running: PID=$pid WID=$(cat "$WINDOW_FILE" 2>/dev/null || echo unknown)"
        else
            echo "Game not running (stale PID file)"
            rm -f "$PID_FILE" "$WINDOW_FILE"
        fi
    else
        echo "No game running"
    fi
}

get_wid() {
    if [[ -f "$WINDOW_FILE" ]]; then
        cat "$WINDOW_FILE"
    else
        find_window || die "No game window found. Run: ./playtest.sh start"
    fi
}

cmd_screenshot() {
    local wid
    wid=$(get_wid)
    local name="${1:-$(date +%s)}"
    local path="$SCREENSHOT_DIR/${name}.png"

    import -window "$wid" "$path" || die "Screenshot failed"
    echo "$path"
}

cmd_key() {
    local wid
    wid=$(get_wid)
    local key="${1:?Usage: playtest.sh key <KeyName>}"
    xdotool key --window "$wid" "$key"
    echo "Sent: $key"
}

cmd_keys() {
    local wid
    wid=$(get_wid)
    local keys="${1:?Usage: playtest.sh keys \"Key1 Key2 ...\"}"
    for k in $keys; do
        xdotool key --window "$wid" "$k"
        sleep 0.1
    done
    echo "Sent: $keys"
}

cmd_play() {
    local wid
    wid=$(get_wid)
    local duration="${1:-3}"
    local directions=(Up Down Left Right)
    local end_time=$((SECONDS + duration))

    echo "Playing for ${duration}s..."
    while (( SECONDS < end_time )); do
        local dir=${directions[$((RANDOM % 4))]}
        xdotool key --window "$wid" "$dir"
        sleep 0.15
    done

    # Take a screenshot at the end
    local path
    path=$(cmd_screenshot "play-$(date +%s)")
    echo "Done. Screenshot: $path"
}

cmd_hold() {
    local wid
    wid=$(get_wid)
    local key="${1:?Usage: playtest.sh hold <Key> <seconds>}"
    local duration="${2:-1}"

    # Simulate holding by sending repeated keydown events
    local end_time=$((SECONDS + duration))
    while (( SECONDS < end_time )); do
        xdotool key --window "$wid" "$key"
        sleep 0.05
    done
    echo "Held $key for ${duration}s"
}

# ── Main dispatch ───────────────────────────────────────────────────

case "${1:-help}" in
    start)      cmd_start ;;
    stop)       cmd_stop ;;
    status)     cmd_status ;;
    screenshot) cmd_screenshot "${2:-}" ;;
    key)        cmd_key "${2:-}" ;;
    keys)       cmd_keys "${2:-}" ;;
    play)       cmd_play "${2:-3}" ;;
    hold)       cmd_hold "${2:-}" "${3:-1}" ;;
    help|*)
        echo "Usage: playtest.sh <command> [args]"
        echo ""
        echo "Commands:"
        echo "  start              Launch game"
        echo "  stop               Kill game"
        echo "  status             Check if running"
        echo "  screenshot [name]  Capture screenshot (prints path)"
        echo "  key <Key>          Send keypress (Left, Right, Up, Down, space)"
        echo "  keys \"K1 K2 ...\"   Send multiple keys"
        echo "  play [seconds]     Random play for N seconds + screenshot"
        echo "  hold <Key> [secs]  Hold a key for N seconds"
        echo ""
        echo "Screenshots saved to: $SCREENSHOT_DIR/"
        echo "View with Claude's Read tool: Read /tmp/pac-playtest/<name>.png"
        ;;
esac
