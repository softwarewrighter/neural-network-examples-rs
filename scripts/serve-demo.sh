#!/usr/bin/env bash

# Serve script for local testing of the GitHub Pages demo
# Serves the docs/ directory on http://127.0.0.1:8080

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$PROJECT_ROOT/docs"
PORT=8080

echo "========================================="
echo "Neural Network Animator Demo Server"
echo "========================================="
echo ""

# Check if docs/ exists
if [ ! -d "$DOCS_DIR" ]; then
    echo "ERROR: docs/ directory not found!"
    echo ""
    echo "Build the demo first with:"
    echo "  bash scripts/build-demo.sh"
    exit 1
fi

# Check for basic-http-server
if command -v basic-http-server &> /dev/null; then
    echo "Serving demo at: http://127.0.0.1:$PORT"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    cd "$DOCS_DIR"
    basic-http-server -a 127.0.0.1:$PORT
elif command -v python3 &> /dev/null; then
    echo "Using Python's HTTP server (basic-http-server not found)"
    echo "Serving demo at: http://127.0.0.1:$PORT"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    cd "$DOCS_DIR"
    python3 -m http.server $PORT
elif command -v python &> /dev/null; then
    echo "Using Python's HTTP server (basic-http-server not found)"
    echo "Serving demo at: http://127.0.0.1:$PORT"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    cd "$DOCS_DIR"
    python -m SimpleHTTPServer $PORT
else
    echo "ERROR: No HTTP server found!"
    echo ""
    echo "Install one of the following:"
    echo "  - basic-http-server: cargo install basic-http-server"
    echo "  - Python 3: Usually pre-installed on macOS/Linux"
    exit 1
fi
