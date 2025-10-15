# Neural Network Animator

An interactive web-based tool for animating neural network training processes. Create educational videos with DVR-like controls (play, pause, step, rewind, speed control) to visualize how networks learn.

## 🚧 Status: Phase 1 Complete - Frontend Pending

**✅ Completed:**
- Animation script format (JSON-based)
- Timeline engine with DVR controls
- Auto-generation from checkpoint files
- CLI tool (generate, validate, serve)
- Test animation (XOR example)
- Comprehensive documentation

**⏳ Pending:**
- Leptos WASM frontend (all-Rust UI)
- SVG network rendering in browser
- Interactive timeline scrubbing
- Real-time playback

The **backend and CLI are fully functional**. You can generate and validate animation scripts. The web UI requires a Leptos implementation (next phase).

## Features

- **Interactive Web UI**: Timeline scrubbing, DVR controls (play/pause, step forward/back, speed control)
- **Auto-generation**: Automatically generate animation scripts from checkpoint files
- **SVG Rendering**: Reuses existing network visualization from `neural-net-viz`
- **Pluggable Scripts**: JSON-based animation format for full control
- **Keyboard Shortcuts**: Space (play/pause), Arrow keys (step), Home/End (jump)
- **Export-Ready**: Perfect for screen capture to create educational videos

## Quick Start

### 1. Generate an Animation Script

Create an animation from existing checkpoints:

```bash
./target/release/neural-net-animator generate \
  --output animation.json \
  --title "XOR Learning" \
  --description "Watch the network learn XOR" \
  --scene-duration 3.0 \
  checkpoints/xor_initial.json \
  checkpoints/xor_trained.json
```

### 2. Start the Server

```bash
./target/release/neural-net-animator serve animation.json --port 8080
```

### 3. Open Browser

Navigate to `http://localhost:8080` and interact with the animation!

## CLI Commands

### `serve` - Start Animation Server

```bash
neural-net-animator serve <SCRIPT> [OPTIONS]

Arguments:
  <SCRIPT>  Path to animation script (JSON)

Options:
  -p, --port <PORT>              Port to serve on [default: 8080]
  --host <HOST>                  Host to bind to [default: 127.0.0.1]
  -c, --checkpoints <DIR>        Checkpoint directory [default: checkpoints]
  -o, --open                     Open browser automatically
```

**Example**:
```bash
./target/release/neural-net-animator serve \
  crates/neural-net-animator/scripts/xor_animation.json \
  --port 8080 \
  --open
```

### `generate` - Create Animation Script

```bash
neural-net-animator generate \
  --output <FILE> \
  [OPTIONS] \
  <CHECKPOINT>...

Arguments:
  <CHECKPOINT>...  Checkpoint files (in chronological order)

Options:
  -o, --output <FILE>                 Output path for script
  -t, --title <TITLE>                 Animation title
  -d, --description <DESC>            Animation description
  --scene-duration <SECS>             Scene duration [default: 2.0]
  --intro-duration <SECS>             Intro/outro duration [default: 3.0]
```

**Example**:
```bash
./target/release/neural-net-animator generate \
  --output xor_animation.json \
  --title "XOR Learning with Backpropagation" \
  --description "Watch how a neural network learns the XOR function" \
  --scene-duration 3.0 \
  --intro-duration 5.0 \
  examples/example-2-backward-propagation-xor/checkpoints/xor_initial.json \
  examples/example-2-backward-propagation-xor/checkpoints/xor_trained.json
```

### `validate` - Validate Animation Script

```bash
neural-net-animator validate <SCRIPT>

Arguments:
  <SCRIPT>  Path to animation script to validate
```

**Example**:
```bash
./target/release/neural-net-animator validate xor_animation.json
```

## Animation Script Format

Animation scripts are JSON files with the following structure:

```json
{
  "metadata": {
    "title": "My Animation",
    "description": "Description here",
    "author": "Author Name",
    "version": "0.1.0"
  },
  "network_info": {
    "architecture": "2-4-1",
    "activation": "Sigmoid",
    "input_size": 2,
    "hidden_size": 4,
    "output_size": 1
  },
  "truth_table": {
    "input_labels": ["A", "B"],
    "output_labels": ["A XOR B"],
    "rows": [
      {"inputs": [0.0, 0.0], "expected": [0.0], "actual": null},
      {"inputs": [0.0, 1.0], "expected": [1.0], "actual": null}
    ]
  },
  "scenes": [
    {
      "id": "intro",
      "duration": 5.0,
      "network_state": {
        "checkpoint_path": "checkpoints/initial.json",
        "iteration": 0,
        "test_results": null,
        "weight_data": null
      },
      "annotations": [
        {
          "annotation_type": "title",
          "text": "Before Training",
          "position": "top",
          "style": {
            "font_size": "32px",
            "color": "#333",
            "weight": "bold"
          }
        }
      ],
      "highlights": [],
      "transition": "morph"
    }
  ]
}
```

### Key Components

- **metadata**: Title, description, author
- **network_info**: Architecture details (auto-extracted from checkpoints)
- **truth_table** (optional): Display expected behavior
- **scenes**: Array of animation scenes

### Scene Structure

Each scene defines:
- **id**: Unique identifier
- **duration**: Length in seconds
- **network_state**: Checkpoint path, iteration, test results
- **annotations**: Text overlays (titles, labels, metrics)
- **highlights**: Visual emphasis (weight changes, neurons, data flow)
- **transition**: How to transition to next scene (cut, fade, morph, slide)

### Annotation Types

- `title`: Large heading text
- `label`: General label
- `metric`: Metric display (accuracy, error, etc.)
- `explanation`: Explanatory text

### Annotation Positions

- Named: `"top"`, `"bottom-left"`, `"center"`, etc.
- Coordinates: `{"x": 100, "y": 200}`

### Highlight Types

- `weight_change`: Highlight weights that changed significantly
- `neurons`: Highlight specific neurons by index
- `data_flow`: Highlight input/output paths

### Transition Types

- `cut`: Instant cut to next scene
- `fade`: Fade out/in
- `morph`: Smoothly interpolate weights
- `slide`: Slide transition

## Web UI Controls

### Playback Controls

- **⏮ Restart**: Jump to beginning
- **◄ Step Back**: Step backward 1 second
- **▶/❚❚ Play/Pause**: Toggle playback
- **► Step Forward**: Step forward 1 second
- **⏭ Jump to End**: Jump to end

### Speed Control

- **0.5× Slower**: Decrease playback speed
- **1× / 2× / 4×**: Current speed
- **2× Faster**: Increase playback speed

Speeds: 0.25×, 0.5×, 1×, 2×, 4×

### Timeline

- **Click timeline**: Scrub to any point
- **Progress bar**: Shows current position

### Keyboard Shortcuts

- **Space**: Play/Pause
- **←**: Step backward
- **→**: Step forward
- **Home**: Jump to start
- **End**: Jump to end

## Creating Animations for Examples

### Step 1: Capture Checkpoints During Training

Modify your example to save intermediate checkpoints:

```rust
// Save initial state
network.save_checkpoint("checkpoints/initial.json", metadata_initial.clone())?;

// Train and save periodically
for epoch in 0..10 {
    // ... training code ...

    if epoch % 100 == 0 {
        let metadata = NetworkMetadata::checkpoint("Network", epoch, Some(accuracy));
        network.save_checkpoint(&format!("checkpoints/epoch_{}.json", epoch), metadata)?;
    }
}

// Save final state
network.save_checkpoint("checkpoints/trained.json", metadata_trained.clone())?;
```

### Step 2: Generate Animation Script

```bash
neural-net-animator generate \
  --output animation.json \
  --title "Your Title" \
  checkpoints/initial.json \
  checkpoints/epoch_100.json \
  checkpoints/epoch_200.json \
  checkpoints/trained.json
```

### Step 3: Customize Script (Optional)

Edit the generated JSON to:
- Add truth tables
- Customize annotations
- Add highlights
- Adjust scene durations

### Step 4: Serve and Test

```bash
neural-net-animator serve animation.json --open
```

### Step 5: Record Video

Use screen recording software (QuickTime, OBS, etc.) to capture the animation.

## Architecture

```
neural-net-animator/
├── src/
│   ├── lib.rs              # Core library
│   ├── script.rs           # Animation script data structures
│   ├── timeline.rs         # Timeline management
│   ├── generator.rs        # Auto-generation from checkpoints
│   └── server/
│       └── mod.rs          # Web server (Axum + embedded HTML)
├── scripts/                # Example animation scripts
│   └── xor_animation.json
└── bin/
    └── main.rs             # CLI binary
```

## Example Workflow

### XOR Animation (Provided)

A complete example is included:

```bash
# Validate it
./target/release/neural-net-animator validate \
  crates/neural-net-animator/scripts/xor_animation.json

# Serve it
./target/release/neural-net-animator serve \
  crates/neural-net-animator/scripts/xor_animation.json \
  --port 8080 \
  --open
```

This animation shows:
1. Initial network with random weights (5 seconds)
2. Trained network after learning XOR (3 seconds)

### Creating More Complex Animations

For examples with many iterations:

```bash
# Generate script with multiple checkpoints
neural-net-animator generate \
  --output optimizer_comparison.json \
  --title "Adam vs SGD" \
  --description "Compare optimizer convergence speeds" \
  --scene-duration 2.0 \
  checkpoints/sgd_initial.json \
  checkpoints/sgd_iter_1000.json \
  checkpoints/sgd_iter_5000.json \
  checkpoints/sgd_trained.json
```

Then manually edit to add:
- Annotations showing iteration count and accuracy
- Highlights showing weight changes
- Truth table for the task

## Future Enhancements

- **Real-time weight interpolation**: Smooth morphing between checkpoints
- **Multiple network comparison**: Side-by-side animations
- **Custom visualizations**: Support for layer-specific views
- **Export frames**: Generate PNG sequence for video editing
- **Annotation editor**: Web-based script editor

## Troubleshooting

### Server won't start

- **Error**: "Address already in use"
  - **Solution**: Use different port: `--port 8081`

### Checkpoint not found

- **Error**: "Failed to load checkpoint"
  - **Solution**: Use absolute paths or ensure checkpoints are relative to current directory

### Animation doesn't play

- **Check**: Browser console (F12) for JavaScript errors
- **Verify**: Script is valid JSON (`validate` command)

## Contributing

This tool is part of the `neural-network-examples-rs` repository. To add features:

1. Modify `src/script.rs` for new data structures
2. Update `src/generator.rs` for auto-generation features
3. Enhance `src/server/mod.rs` for UI improvements

## License

MIT - See repository LICENSE file

---

**Tip**: For best results, use 0.5× speed when recording tutorials, then speed up in video editing for smooth narration.
