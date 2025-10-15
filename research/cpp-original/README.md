# Original C++ Implementation

This directory contains the original C++ implementation of the Feed-Forward Neural Network that served as the reference for the Rust port.

## Source

**Repository**: [Feed-Forward-Neural-Network](https://github.com/wimacod/Feed-Forward-Neural-Network)
**Author**: Alexis Louis
**Date**: March 2016
**License**: MIT (see LICENSE in root)

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `FFN.hpp/cpp` | Feed-Forward Network class | ~150 |
| `Layer.hpp/cpp` | Layer implementation (forward/backward prop) | ~200 |
| `main.cpp` | Example usage (XOR, digit recognition) | ~35 |
| `graphic_f.hpp/cpp` | SFML visualization | ~100 |
| `utility_f.hpp/cpp` | File I/O utilities | ~50 |
| `Header.h` | Common includes | ~10 |

## Key Features

- 3-layer feed-forward network (input → hidden → output)
- Sigmoid activation function
- Backpropagation with gradient descent
- Training by iteration or error threshold
- SFML-based visualization
- XOR and digit recognition examples

## Architecture Notes

### Circular Dependencies

The C++ implementation uses circular dependencies between `FFN` and `Layer`:

```cpp
class FFN {
    vector<Layer*> layers;  // FFN owns layers
};

class Layer {
    FFN *network;  // Layer has back-pointer to network
};
```

This pattern is common in C++ but problematic in Rust. The Rust port eliminates this by:
1. Separating data structures from algorithms (extension traits)
2. Using borrowing instead of ownership for temporary access
3. Avoiding back-references entirely

### Memory Management

The original uses raw pointers and manual memory management:

```cpp
FFN *XORnetwork = new FFN();
// ... use network ...
// No explicit delete (potential memory leak)
```

The Rust port uses ownership and RAII:

```rust
let mut network = FeedForwardNetwork::new(2, 4, 1);
// Automatically dropped when out of scope
```

### Platform Dependencies

The C++ version requires:
- SFML library (for graphics)
- Platform-specific compilation
- External dependencies

The Rust version is pure Rust (except ndarray):
- SVG generation (no external library)
- Platform-independent
- Minimal dependencies

## Mapping to Rust

See `docs/history.md` for detailed mapping of C++ files to Rust equivalents.

Quick reference:

| C++ | Rust |
|-----|------|
| `FFN.cpp` | `crates/neural-net-types/src/network.rs` + `crates/neural-net-core/src/forward.rs` + `crates/neural-net-core/src/backward.rs` |
| `Layer.cpp` | `crates/neural-net-types/src/layer.rs` + algorithms in `neural-net-core` |
| `main.cpp` | `examples/example-2-backward-propagation-*/src/main.rs` |
| `graphic_f.cpp` | `crates/neural-net-viz/src/lib.rs` |
| `utility_f.cpp` | `crates/neural-net-core/src/utils/file_io.rs` |

## What Was Ported

✅ All core functionality:
- Forward propagation
- Backpropagation
- Weight updates
- Training algorithms
- XOR example

✅ Plus improvements:
- Modern optimizers (Adam, RMSprop, AdamW)
- Error handling (Result types)
- Serialization (JSON checkpoints)
- Comprehensive tests (81+ tests)
- More examples (8 vs 1)
- Platform-independent visualization (SVG)

## What Was NOT Ported

❌ **Digit recognition example**: Not yet implemented (planned for future)
❌ **SFML visualization**: Replaced with SVG generation (no external deps)

## References

For detailed information about the port, see:
- `docs/history.md` - Complete C++ to Rust migration documentation
- `docs/architecture.md` - Rust architecture and design decisions
- `docs/PRD.md` - Project goals and roadmap

## Usage

These files are for **reference only**. They are not compiled or used in the Rust project.

To see the original in action:
```bash
# (In a separate directory)
git clone https://github.com/wimacod/Feed-Forward-Neural-Network
cd Feed-Forward-Neural-Network
# Follow the C++ compilation instructions in that repo
```

---

**Note**: These files are preserved for historical context and learning purposes. The Rust implementation is the active, maintained version.
