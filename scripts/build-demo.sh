#!/usr/bin/env bash

# Build script for GitHub Pages demo
# Compiles the Yew/WASM app and copies demo assets to docs/ directory

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_DIR="$PROJECT_ROOT/crates/neural-net-animator/web"
DOCS_DIR="$PROJECT_ROOT/docs"

echo "========================================="
echo "Building Neural Network Animator Demo"
echo "========================================="
echo ""

# Step 1: Check for trunk
echo "[1/5] Checking for trunk..."
if ! command -v trunk &> /dev/null; then
    echo "ERROR: trunk is not installed!"
    echo "Install with: cargo install trunk"
    exit 1
fi
echo "✓ trunk found"
echo ""

# Step 2: Clean previous build
echo "[2/5] Cleaning previous build..."
rm -rf "$DOCS_DIR"
mkdir -p "$DOCS_DIR"
echo "✓ Cleaned $DOCS_DIR"
echo ""

# Step 3: Build WASM app with Trunk
echo "[3/5] Building WASM app with trunk..."
cd "$WEB_DIR"
trunk build --release --dist "$DOCS_DIR"
echo "✓ WASM app built"
echo ""

# Step 4: Copy demo assets (animation scripts and checkpoints)
echo "[4/5] Copying demo assets..."

# Create assets directory structure
mkdir -p "$DOCS_DIR/examples/example-2-backward-propagation-xor/checkpoints"
mkdir -p "$DOCS_DIR/scripts"

# Copy animation script
cp "$PROJECT_ROOT/crates/neural-net-animator/scripts/xor_animation.json" \
   "$DOCS_DIR/scripts/xor_animation.json"
echo "  ✓ Copied animation script"

# Copy XOR checkpoints
cp "$PROJECT_ROOT/examples/example-2-backward-propagation-xor/checkpoints"/*.json \
   "$DOCS_DIR/examples/example-2-backward-propagation-xor/checkpoints/"
echo "  ✓ Copied XOR checkpoints"

echo "✓ Demo assets copied"
echo ""

# Step 5: Create .nojekyll file (required for GitHub Pages)
echo "[5/5] Creating .nojekyll file..."
touch "$DOCS_DIR/.nojekyll"
echo "✓ Created .nojekyll"
echo ""

# Summary
echo "========================================="
echo "Build Complete!"
echo "========================================="
echo ""
echo "Output directory: $DOCS_DIR"
echo ""
echo "To test locally, run:"
echo "  bash scripts/serve-demo.sh"
echo ""
echo "To deploy to GitHub Pages:"
echo "  1. Commit the docs/ directory"
echo "  2. Push to GitHub"
echo "  3. Enable GitHub Pages in repository settings"
echo "     (Source: Deploy from a branch -> main -> /docs)"
echo ""
