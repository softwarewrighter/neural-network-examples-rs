# Architecture - Feed-Forward Neural Network in Rust

## Overview

This document describes the architecture for a Rust implementation of a multi-layer feed-forward neural network (FFN) with backpropagation. The design prioritizes type safety, zero-cost abstractions, and idiomatic Rust patterns while maintaining the mathematical clarity of the original C++ implementation.

## Core Components

### 1. Network Structure

```
┌─────────────────────────────────────────────────────┐
│                   FeedForwardNetwork                 │
│  - layers: Vec<Layer>                                │
│  - targets: Option<Vec<f32>>                         │
│                                                       │
│  + new() -> Self                                     │
│  + init(input_size, hidden_size, output_size)       │
│  + forward(&self, inputs: &[f32]) -> Vec<f32>       │
│  + train_by_iteration(...)                          │
│  + train_by_error(...)                              │
│  + test(...)                                         │
└─────────────────────────────────────────────────────┘
                          │
                          │ contains
                          ▼
         ┌────────────────────────────────┐
         │           Layer                │
         │  - index: usize                │
         │  - num_neurons: usize          │
         │  - weights: Option<Array2<f32>>│
         │  - inputs: Vec<f32>            │
         │  - outputs: Vec<f32>           │
         │  - deltas: Vec<f32>            │
         │                                │
         │  + forward_propagate(...)      │
         │  + backward_propagate(...)     │
         │  + update_weights(...)         │
         └────────────────────────────────┘
```

### 2. Module Organization

```
src/
├── lib.rs                  # Library root, re-exports public API
├── network.rs              # FeedForwardNetwork implementation
├── layer.rs                # Layer implementation
├── activation.rs           # Activation functions (sigmoid, linear, etc.)
├── trainer.rs              # Training algorithms and strategies
├── testing.rs              # Testing and evaluation utilities
└── utils/
    ├── mod.rs             # Utilities module root
    ├── file_io.rs         # Matrix file reading/writing
    └── visualization.rs   # Optional: plotting/visualization
examples/
├── xor.rs                 # XOR learning example
└── digit_recognition.rs   # Digit recognition example
tests/
├── integration_tests.rs   # End-to-end tests
└── unit_tests.rs          # Component tests
```

## Key Design Decisions

### 1. Type System & Memory Safety

**Array Representation:**
- Use `ndarray` crate for weight matrices (provides BLAS/LAPACK integration)
- Use `Vec<f32>` for layer inputs/outputs/deltas (simpler, sufficient for 1D)
- Benefits: Type safety, no manual memory management, optimized linear algebra

**Ownership Model:**
```rust
pub struct FeedForwardNetwork {
    layers: Vec<Layer>,           // Owned layers
    targets: Option<Vec<f32>>,    // Optional current targets
}

pub struct Layer {
    index: usize,
    num_neurons: usize,
    weights: Option<Array2<f32>>, // None for input layer
    inputs: Vec<f32>,
    outputs: Vec<f32>,
    deltas: Vec<f32>,
}
```

- Layers are owned by the network (no circular references)
- No raw pointers or unsafe code needed
- Clear ownership hierarchy: Network → Layers → Data

### 2. Activation Functions

Trait-based design for extensibility:

```rust
pub trait Activation {
    fn activate(&self, x: f32) -> f32;
    fn derivative(&self, output: f32) -> f32;
}

pub struct Sigmoid;
pub struct Linear;
pub struct ReLU;  // Future extension
pub struct Tanh;  // Future extension
```

### 3. Training Strategy

Builder pattern for training configuration:

```rust
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub stopping_criteria: StoppingCriteria,
}

pub enum StoppingCriteria {
    ByIteration(usize),
    ByError(f32),
    ByTime(Duration),
}
```

### 4. Error Handling

Use `Result<T, NeuralNetError>` for fallible operations:

```rust
#[derive(Debug, thiserror::Error)]
pub enum NeuralNetError {
    #[error("Invalid layer configuration: {0}")]
    InvalidConfig(String),

    #[error("Input dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Training failed: {0}")]
    TrainingError(String),
}
```

## Algorithm Details

### Forward Propagation

1. **Input Layer** (index 0): `outputs = inputs` (linear passthrough)
2. **Hidden Layers** (index 1..n-1): `outputs[i] = sigmoid(Σ(weights[j][i] * prev_outputs[j]))`
3. **Output Layer** (index n): `outputs = inputs` (linear for regression/classification)

### Backpropagation

1. **Output Layer**: `δ[i] = (target[i] - output[i])`
2. **Hidden Layers**: `δ[i] = Σ(next_weights[i][j] * next_δ[j]) * output[i] * (1 - output[i])`
3. **Weight Update**: `w[i][j] += η * δ[j] * prev_output[i]`

Where:
- `η` (eta) = learning rate (default 0.01)
- `δ` (delta) = error gradient

### Random Initialization

Use `rand` crate with proper seeding:
- Weights initialized uniformly in range [-1.0, 1.0]
- Reproducible via seed option for testing

## Dependencies

```toml
[dependencies]
ndarray = "0.15"           # Multi-dimensional arrays
rand = "0.8"               # Random number generation
thiserror = "1.0"          # Error type derivation

[dev-dependencies]
approx = "0.5"             # Float comparison in tests
criterion = "0.5"          # Benchmarking
```

## Performance Considerations

### Optimizations

1. **BLAS Integration**: `ndarray` can use optimized BLAS libraries
2. **Memory Layout**: Contiguous memory for cache efficiency
3. **Borrowing**: Use references to avoid unnecessary copies
4. **Iterator Chains**: Leverage lazy evaluation

### Parallelization (Future)

- Use `rayon` for parallel batch training
- SIMD operations via `ndarray` + BLAS
- GPU acceleration via `wgpu` or `cuda` (optional)

## Differences from C++ Implementation

| Aspect | C++ | Rust |
|--------|-----|------|
| Memory | Manual pointers, new/delete | Automatic (ownership) |
| Arrays | std::vector | ndarray::Array2 + Vec |
| Errors | Unchecked / exceptions | Result<T, E> |
| Random | std::random_device | rand crate |
| Generics | Templates | Traits + generics |
| Network-Layer relationship | Bidirectional pointers | Unidirectional ownership |

## Testing Strategy

### Unit Tests
- Activation function correctness
- Matrix operations
- Forward/backward propagation on simple networks

### Integration Tests
- XOR learning (converges to correct solution)
- Digit recognition (achieves >90% accuracy)
- Serialization/deserialization

### Property-Based Tests (using `proptest`)
- Gradient checking (numerical vs analytical)
- Weight updates preserve dimensions
- Network output stability

## Future Enhancements

1. **Multiple Hidden Layers**: Generalize from 3-layer to n-layer
2. **Activation Variants**: ReLU, Leaky ReLU, Tanh, etc.
3. **Optimization Algorithms**: Momentum, Adam, RMSprop
4. **Regularization**: L1/L2, dropout
5. **Mini-batch Training**: Stochastic gradient descent
6. **Serialization**: Save/load trained models (serde)
7. **CLI Tool**: Command-line interface for training
8. **GPU Support**: CUDA/OpenCL via compute shaders
