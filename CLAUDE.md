# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **educational ML demonstration platform** built in Rust, starting with a feed-forward neural network port from C++. The goal is to create a comprehensive learning resource with:

- **Reusable core library** (`src/`) with type-safe ML components
- **Incremental examples** (future: `examples/01-feedforward/`, `02-optimizers/`, etc.)
- **Tutorial-oriented** code with visualizations and benchmarks

**Current Status:** v0.1 - Phase 0 complete (foundation), implementing forward propagation next.

## Essential Commands

### Build & Test
```bash
cargo build                     # Build the project
cargo test                      # Run all tests
cargo test -- --nocapture       # Run tests with output
cargo test test_name            # Run specific test
cargo test --lib                # Run library tests only
cargo test --doc                # Run doc tests only
```

### Code Quality (Local CI)
```bash
cargo clippy -- -D warnings     # Lint (zero warnings required)
cargo fmt                       # Auto-format code
cargo fmt -- --check            # Check formatting without modifying
cargo doc --open                # Build and view documentation
cargo doc --no-deps             # Build docs (library only)
```

### Future Commands (when examples exist)
```bash
cargo run --example xor         # Run XOR example
cargo run --example digit_recognition  # Run digit recognition
cargo bench                     # Run benchmarks (Phase 5+)
```

## Architecture & Design Patterns

### Core Components Hierarchy

```
FeedForwardNetwork (src/network.rs)
  ├── owns: Vec<Layer> (src/layer.rs)
  │   └── owns: Option<Array2<f32>> weights (ndarray)
  ├── uses: Activation trait (src/activation.rs)
  └── returns: Result<T, NeuralNetError> (src/error.rs)
```

**Key Design Principle:** Network owns Layers (unidirectional ownership), eliminating C++'s circular reference pattern.

### Activation Functions (Trait-Based)

Located in `src/activation.rs`:
```rust
pub trait Activation {
    fn activate(&self, x: f32) -> f32;
    fn derivative(&self, output: f32) -> f32;
}
```

Implementations: `Sigmoid`, `Linear` (ReLU, Tanh planned).

**Pattern:** All extensible components use traits (future: optimizers, losses, layers).

### Error Handling Strategy

All fallible operations return `Result<T, NeuralNetError>`. Never panic in library code.

Error types in `src/error.rs`:
- `InvalidConfig` - Layer/network setup issues
- `DimensionMismatch` - Input size validation failures
- `IoError` - File reading errors
- `TrainingError` - Training failures

**Pattern:** Use `?` operator to propagate errors; provide context in error messages.

### Layer Implementation Details

Layers store:
- `index: usize` - Position in network (0=input, 1=hidden, 2=output for v0.1)
- `weights: Option<Array2<f32>>` - None for input layer, Some for hidden/output
- `inputs/outputs/deltas: Vec<f32>` - Working memory for forward/backward pass

**Critical:** Input layer (index 0) has no weights; hidden/output layers initialize weights in [-1.0, 1.0].

### Neural Network Algorithm Flow

**Forward Propagation (Phase 2, TODO):**
1. Input layer: `outputs = inputs` (linear passthrough)
2. Hidden layer: `outputs[i] = sigmoid(Σ(weights[j][i] * prev_outputs[j]))`
3. Output layer: `outputs = inputs` (linear)

**Backpropagation (Phase 3, TODO):**
1. Output layer: `δ[i] = (target[i] - output[i])`
2. Hidden layer: `δ[i] = Σ(next_weights[i][j] * next_δ[j]) * output[i] * (1 - output[i])`
3. Weight update: `w[i][j] += η * δ[j] * prev_output[i]` (η=0.01 default)

## Development Workflow

### Test-Driven Development (TDD)

**Required approach:** Red-Green-Refactor for all new features.

1. **Red:** Write failing test defining desired behavior
2. **Green:** Implement minimal code to pass test
3. **Refactor:** Improve code while keeping tests green

**Example workflow for implementing forward propagation:**
```rust
// 1. RED: Write test first
#[test]
fn test_forward_pass_known_weights() {
    let mut network = FeedForwardNetwork::new(2, 2, 1);
    // Set known weights, test against manual calculation
    let output = network.forward(&[1.0, 0.5]).unwrap();
    assert_relative_eq!(output[0], 0.731, epsilon = 0.001);
}

// 2. GREEN: Implement Layer::calc_inputs(), calc_outputs()
// 3. REFACTOR: Optimize with BLAS if needed
```

### Code Quality Standards

- **Zero clippy warnings** (enforced in local CI)
- **Consistent formatting** (rustfmt)
- **100% public API documentation** (rustdoc with examples)
- **No `unsafe` code** in library (Phase 1-4)
- **Result-based error handling** (no panics)

### File Organization for Future Phases

When adding new features, follow this structure:
- **Core algorithms:** `src/` (e.g., `src/layers/conv.rs` for CNNs)
- **Examples:** `examples/XX-topic/` with README, code, visualizations
- **Tests:** `tests/` for integration, `src/` for unit tests
- **Docs:** `docs/` for architecture/planning, rustdoc for API

## Important Implementation Notes

### ndarray Usage

Weight matrices use `ndarray::Array2<f32>`:
```rust
// Create: Array2::from_shape_fn((rows, cols), |_| value)
// Access: weights[[row, col]]
// Multiply: Use dot(), or manual loops in Phase 2
```

**BLAS integration** available but not required initially.

### Layer-Network Relationship (Key Difference from C++)

C++ used bidirectional pointers (`Layer* → FFN*`, `FFN* → Layer*`). Rust uses unidirectional ownership:

```rust
// Network owns layers
pub struct FeedForwardNetwork {
    layers: Vec<Layer>,  // Owned
}

// Layers don't reference network
// Pass context via function parameters instead:
layer.forward_propagate(prev_outputs, is_output_layer)?;
```

**Pattern:** Use `split_at_mut()` when need to access adjacent layers during backprop.

### Weight Initialization

Weights initialized uniformly in `[-1.0, 1.0]` using `rand::thread_rng()`:
```rust
Array2::from_shape_fn((prev_size, num_neurons), |_| {
    rng.gen_range(-1.0..1.0)
})
```

**Testing:** Deterministic initialization (seeded RNG) not yet implemented but planned.

### Phase Implementation Status

- ✅ **Phase 0:** Project setup, error types, activation traits, skeleton structures
- ⏳ **Phase 2:** Forward propagation (current focus)
- ⏳ **Phase 3:** Backpropagation and training
- ⏳ **Phase 4:** Testing utilities, examples (XOR, digit recognition)
- ⏳ **Phase 5:** Documentation, benchmarks
- ⏳ **Phase 6:** Release preparation

See `docs/plan.md` for detailed phase breakdown.

## Documentation References

**Before implementing features, read:**
- `docs/architecture.md` - Technical architecture, design decisions, algorithms
- `docs/plan.md` - 6-phase implementation plan with code snippets and C++ mappings
- `docs/PRD.md` - Requirements, success metrics, future roadmap
- `docs/learnings.md` - Key decisions (TDD, local CI, project vision)

**For specific topics:**
- Error types: `src/error.rs`
- Activation functions: `src/activation.rs` (trait pattern reference)
- Network structure: `docs/architecture.md` sections 1-2
- Algorithms: `docs/architecture.md` "Algorithm Details"

## Common Patterns

### Adding New Activation Function
1. Implement `Activation` trait in `src/activation.rs`
2. Add tests for `activate()` and `derivative()`
3. Export from `src/lib.rs`

### Creating Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;  // For float comparison

    #[test]
    fn test_feature() {
        // Arrange, Act, Assert
    }
}
```

### Error Propagation
```rust
pub fn fallible_operation(&self) -> Result<Output> {
    let data = self.load_data()?;  // Propagate with ?
    if data.is_empty() {
        return Err(NeuralNetError::InvalidConfig("Empty data".into()));
    }
    Ok(process(data))
}
```

## Files to Never Modify

- `research/` - Transient artifacts (gitignored), reference materials
- `samples/` - Training data (Xapp.txt, TA.txt, Xtest.txt, TT.txt)
- `Cargo.lock` - Committed for binary crates, gitignored for libraries

## Phase-Specific Context

**Current Phase (2):** Implementing forward propagation in `src/layer.rs` and `src/network.rs`.

**Key tasks:**
1. Complete `Layer::calc_inputs()` - matrix-vector multiplication
2. Complete `Layer::calc_outputs()` - apply activation functions
3. Complete `FeedForwardNetwork::forward()` - orchestrate layer propagation
4. Write tests with known input/output pairs (TDD)

**Reference implementation:** C++ `Layer.cpp:41-70` for `calc_inputs`/`calc_outputs`, `FFN.cpp:24-29` for forward pass.

**Success criteria:** Tests pass, clippy clean, forward pass produces expected outputs for simple 2-input network.
