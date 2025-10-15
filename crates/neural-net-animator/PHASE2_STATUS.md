# Phase 2 Status: Leptos Frontend (In Progress)

## Current Status

Phase 2 implementation of the Leptos WASM frontend has been started but **does not compile yet** due to Leptos 0.7 API compatibility issues.

## What's Been Created

### ✅ Project Structure
- `web/` directory with proper Cargo.toml
- Trunk configuration (Trunk.toml, index.html, styles.css)
- All component files created with logical structure

### ✅ Components Created (need API fixes)
1. **AnimationPlayer** (src/components/animation_player.rs)
   - Main coordinator component
   - Loads animation script from server
   - Manages timeline state
   - Coordinates child components

2. **NetworkCanvas** (src/components/network_canvas.rs)
   - Displays SVG network visualization
   - Simple component, minimal issues

3. **DvrControls** (src/components/dvr_controls.rs)
   - Playback control buttons
   - Speed controls
   - Timeline state management

4. **Timeline** (src/components/timeline.rs)
   - Interactive progress bar
   - Time display
   - Scrubbing functionality

5. **MetricsPanel** (src/components/metrics_panel.rs)
   - Displays accuracy, error, test results
   - Per-example breakdown

6. **InfoPanel** (src/components/info_panel.rs)
   - Shows scene information
   - Annotations display
   - Network architecture info

### ✅ Server Updates
- Modified server to serve Trunk-built WASM files
- Added fallback_service for static files
- Updated checkpoint handler for full path support
- Placeholder handler when Trunk hasn't built yet

### ✅ Styling
- Complete CSS file (styles.css) with:
  - Responsive layout
  - DVR-style controls
  - Timeline scrubber
  - Metrics panels

## Issues to Resolve

###  1. Leptos 0.7 API Changes

The code was initially written assuming Leptos 0.6 patterns. Leptos 0.7 has breaking changes:

**Signal API Changes:**
```rust
// Old (doesn't work):
set_signal(value);

// New (Leptos 0.7):
set_signal.set(value);
```

**spawn_local:**
```rust
// Need to import:
use leptos::task::spawn_local;
```

**wasm_bindgen:**
```rust
// Old:
use leptos::prelude::wasm_bindgen::JsCast;

// New:
use wasm_bindgen::JsCast;
```

**IntoView/Render traits:**
```rust
// String references need .to_string() in views
// Old: {&script.metadata.title}
// New: {script.metadata.title.clone()}
```

### 2. Compilation Errors Summary

Current error count: ~20-30 errors related to:
- Signal setter syntax (`.set()` vs direct call)
- Missing imports (`spawn_local`, proper JsCast import)
- String rendering in views (need `.clone()` or `.to_string()`)
- Interval `.clear()` method (gloo-timers API)

## How to Fix

### Step 1: Update Signal Usage

Search and replace throughout `web/src/`:

```rust
// Pattern 1: set_signal(value) → set_signal.set(value)
set_script_data(Some(script))  →  set_script_data.set(Some(script))
set_loading(false)  →  set_loading.set(false)
set_error(Some(e))  →  set_error.set(Some(e))

// Pattern 2: signal() → signal.get()
if loading()  →  if loading.get()
timeline()  →  timeline.get()
```

### Step 2: Fix Imports

In `animation_player.rs`:
```rust
use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::JsCast;  // NOT leptos::prelude::wasm_bindgen
```

### Step 3: Fix String Rendering

In all view! macros, clone strings:
```rust
// Before:
<h2>{&script.metadata.title}</h2>

// After:
<h2>{script.metadata.title.clone()}</h2>
```

### Step 4: Fix Interval Cleanup

In `animation_player.rs`:
```rust
// Old:
if let Ok(h) = handle {
    h.clear();
}

// New (check gloo-timers docs):
// Interval may not have .clear(), might auto-drop or need different pattern
```

## Next Steps (Priority Order)

1. **Fix Signal API** (highest priority - affects everything)
   - Update all `set_signal(value)` to `set_signal.set(value)`
   - Update all `signal()` to `signal.get()` where appropriate

2. **Fix Imports**
   - Add `leptos::task::spawn_local`
   - Remove incorrect wasm_bindgen path
   - Add missing web-sys features if needed

3. **Fix String Rendering**
   - Clone all string references in views
   - Use `.to_string()` where appropriate

4. **Test Compilation**
   - `cargo check -p neural-net-animator-web`
   - Fix remaining errors

5. **Build with Trunk**
   - `cd crates/neural-net-animator/web && trunk build`
   - Test in browser

6. **Integrate SVG Rendering**
   - Currently using placeholder text
   - Need to actually load checkpoint and generate SVG using neural-net-viz

## Estimated Effort

- Fixing Leptos 0.7 API issues: **2-4 hours** (systematic find-and-replace plus testing)
- SVG rendering integration: **1-2 hours**
- Testing and refinement: **1-2 hours**

**Total: 4-8 hours of focused work**

## Alternative Approach

If Leptos 0.7 continues to be problematic, consider:

1. **Downgrade to Leptos 0.6** - Use known-working API
2. **Switch to Yew** - More stable, mature framework
3. **Use vanilla WASM-bindgen** - More control, less framework magic
4. **Use Dioxus** - Alternative Rust UI framework

## Reference Resources

- Leptos 0.7 Migration Guide: https://leptos-rs.github.io/leptos/
- Leptos Book: https://book.leptos.dev/
- Example apps: https://github.com/leptos-rs/leptos/tree/main/examples

## Testing Once Fixed

```bash
# 1. Build the web frontend
cd crates/neural-net-animator/web
trunk build --release

# 2. Run the server
cd ../../..
cargo run --release --bin neural-net-animator -- serve \
  crates/neural-net-animator/scripts/xor_animation.json

# 3. Open browser
open http://localhost:8080
```

## Summary

**Phase 1 (Complete):** ✅ Solid backend with CLI, script format, timeline engine
**Phase 2 (Partial):** ⚠️ Frontend structure created but needs Leptos 0.7 API fixes

The architecture is sound, components are well-designed, and styling is complete. The remaining work is primarily mechanical API updates to match Leptos 0.7's patterns.
