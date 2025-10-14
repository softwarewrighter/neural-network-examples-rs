# Architecture - Neural Network Platform in Rust

## Overview

This document describes the architecture for a comprehensive neural network platform in Rust, beginning with a multi-layer feed-forward network (FFN) implementation. The design prioritizes type safety, zero-cost abstractions, and idiomatic Rust patterns while maintaining mathematical clarity.

**Project Vision:** Create an educational ML platform with a reusable core library supporting diverse neural network architectures. Each technique/concept lives in its own example directory with tutorials and visualizations.

**Current Phase (v0.1):** Implement the foundational feed-forward network as the basis for future work.

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

#### Current Structure (v0.1)

```
neural-network-rs/
├── src/                    # Core reusable library
│   ├── lib.rs             # Library root, re-exports public API
│   ├── error.rs           # Error types (NeuralNetError, Result)
│   ├── activation.rs      # Activation functions (trait-based)
│   ├── layer.rs           # Layer implementation
│   ├── network.rs         # FeedForwardNetwork implementation
│   └── utils/
│       ├── mod.rs         # Utilities module root
│       └── file_io.rs     # Matrix file reading/writing
├── examples/
│   ├── xor.rs             # XOR learning example
│   └── digit_recognition.rs # Digit recognition example
├── tests/
│   └── integration_tests.rs # End-to-end tests
├── benches/
│   └── training_benchmark.rs # Performance benchmarks
├── docs/
│   ├── architecture.md    # This document
│   ├── PRD.md            # Product requirements
│   ├── plan.md           # Implementation plan
│   └── learnings.md      # Decisions and lessons learned
└── samples/               # Training/test data
    ├── Xapp.txt
    ├── TA.txt
    ├── Xtest.txt
    └── TT.txt
```

#### Future Structure (v0.2+)

As the project evolves, examples will be organized by topic:

```
neural-network-rs/
├── src/                    # Core library (shared by all examples)
│   ├── lib.rs
│   ├── layers/            # Layer types (Dense, Conv2D, LSTM, etc.)
│   ├── optimizers/        # SGD, Adam, RMSprop, etc.
│   ├── losses/            # MSE, CrossEntropy, etc.
│   ├── activations/       # Sigmoid, ReLU, Tanh, etc.
│   └── utils/
├── examples/
│   ├── 01-feedforward/    # Basic FFN (v0.1)
│   │   ├── README.md      # Tutorial on FFN concepts
│   │   ├── xor.rs
│   │   ├── digit_recognition.rs
│   │   └── visualize.rs   # Decision boundary plots
│   ├── 02-optimizers/     # Advanced training (v0.2)
│   │   ├── README.md      # Tutorial on optimizers
│   │   ├── sgd_momentum.rs
│   │   ├── adam.rs
│   │   └── compare.rs     # Benchmark different optimizers
│   ├── 03-regularization/ # Overfitting prevention (v0.3)
│   │   ├── README.md
│   │   ├── l1_l2.rs
│   │   └── dropout.rs
│   ├── 04-cnn/           # Convolutional networks (v0.4)
│   │   ├── README.md
│   │   ├── mnist.rs
│   │   └── cifar10.rs
│   └── ...               # More as project evolves
├── docs/
│   ├── architecture.md
│   ├── PRD.md
│   ├── plan.md
│   ├── learnings.md
│   └── tutorials/        # In-depth learning guides
└── tests/
    ├── integration/      # Full pipeline tests
    └── benchmarks/       # Performance comparisons
```

**Design Principle:** Each `examples/XX-topic/` directory is self-contained with its own README, but relies on the shared `src/` library for core functionality. This allows:
- Incremental learning (01 → 02 → 03...)
- Code reuse across examples
- Clear separation of concepts
- Easy addition of new techniques

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

## Development Approach

### Test-Driven Development (TDD)

**Philosophy:** Write tests first, then implement to pass tests (Red-Green-Refactor).

**Benefits:**
- Clear behavior specification before coding
- High test coverage by design
- Confidence when refactoring
- Better API design (tests expose issues early)

**Example workflow for Phase 2 (Forward Propagation):**
1. **Red:** Write test with known input/output (e.g., 2-layer network, specific weights, expected output)
2. **Green:** Implement `Layer::calc_inputs()` and `Layer::calc_outputs()` to pass test
3. **Refactor:** Optimize for performance (BLAS, inline functions)
4. Repeat for edge cases (dimension mismatches, extreme values)

### Continuous Integration

**Local CI:** Project uses local servers for CI, not GitHub Actions.

**Rationale:** Full control, no quotas, faster feedback, privacy for proprietary work.

**CI Commands:**
```bash
cargo test              # All tests
cargo clippy -- -D warnings  # Strict linting
cargo fmt -- --check    # Format verification
cargo doc --no-deps     # Documentation build
cargo bench             # Benchmarks (Phase 5+)
```

## Future Enhancements

### v0.2+ Roadmap

Each future release adds new techniques as example directories:

1. **examples/02-optimizers/** (v0.2)
   - SGD with momentum
   - Adam optimizer
   - Learning rate schedules
   - Comparative benchmarks

2. **examples/03-regularization/** (v0.3)
   - L1/L2 regularization
   - Dropout
   - Early stopping
   - Overfitting demonstrations

3. **examples/04-cnn/** (v0.4)
   - Convolutional layers in `src/layers/conv.rs`
   - Pooling layers
   - Image classification (MNIST, CIFAR-10)
   - Feature visualization

4. **examples/05-rnn/** (v0.5)
   - LSTM/GRU layers in `src/layers/recurrent.rs`
   - Time series prediction
   - Sequence-to-sequence models

5. **examples/06-gan/** (v0.6)
   - Generative adversarial networks
   - Image generation
   - Style transfer

6. **examples/07-transformers/** (v0.7)
   - Attention mechanisms
   - Multi-head attention
   - Positional encoding

### Core Library Enhancements

As examples are added, the core library (`src/`) grows:
- **Multiple Hidden Layers**: Generalize from 3-layer to n-layer
- **Layer Types**: Dense, Conv2D, MaxPool, LSTM, Attention, etc.
- **Optimizers**: SGD, Momentum, Adam, RMSprop, AdaGrad
- **Loss Functions**: MSE, CrossEntropy, Hinge, etc.
- **Activation Variants**: ReLU, Leaky ReLU, Tanh, Swish, etc.
- **Serialization**: Save/load trained models (serde)
- **GPU Support**: CUDA/OpenCL via compute shaders (optional)

### Tooling

- **CLI Tool**: Command-line interface for training networks
- **Visualization**: Real-time training dashboards
- **Data Utilities**: Dataset loading, augmentation, preprocessing
- **Model Export**: ONNX format for interoperability
