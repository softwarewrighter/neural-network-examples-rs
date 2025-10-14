# Product Requirements Document - Neural Network Library (Rust)

## Executive Summary

This project creates a comprehensive machine learning demonstration platform in Rust, starting with a port of the existing C++ feed-forward neural network. The initial release (v0.1) will establish a reusable core library with type-safe, performant implementations of fundamental ML concepts.

**Long-Term Vision:** Build an educational ML platform showcasing diverse neural network architectures and techniques through incremental examples (feedforward → CNNs → RNNs → GANs → transformers, etc.). Each example includes tutorials, visualizations, and working code.

**Current Focus (v0.1):** Port and enhance the C++ feedforward network implementation as the foundation for future work.

**Project Name:** `neural-network-rs`
**Version:** 0.1.0
**Target Audience:** ML practitioners, students, researchers, Rust developers
**Platform:** Cross-platform (Linux, macOS, Windows)

## Goals & Objectives

### Primary Goals (v0.1)

1. **Feature Parity**: Replicate all functionality from the C++ implementation
2. **Safety**: Eliminate memory safety issues through Rust's ownership system
3. **Performance**: Match or exceed C++ performance through zero-cost abstractions
4. **Usability**: Provide an ergonomic, well-documented API
5. **Extensibility**: Design for future enhancements (new layers, optimizers, etc.)

### Long-Term Goals (v0.2+)

1. **Core Library**: Reusable data structures and algorithms for diverse ML architectures
2. **Example Collection**: Incremental learning path from basic FFN to advanced techniques
3. **Educational Resource**: Comprehensive tutorials, visualizations, and documentation
4. **Community Platform**: Foster learning and contribution around ML in Rust

### Success Metrics

- ✓ XOR learning converges within 5000 iterations
- ✓ Digit recognition achieves ≥90% accuracy on test set
- ✓ Training performance within 10% of C++ implementation
- ✓ Zero memory leaks or safety issues (guaranteed by Rust)
- ✓ Comprehensive documentation (100% public API coverage)
- ✓ Test coverage ≥80%

## Functional Requirements

### FR-1: Network Construction

**Priority:** P0 (Must Have)

- **FR-1.1**: Create a feed-forward network with configurable layer sizes
- **FR-1.2**: Support 3-layer architecture (input, hidden, output)
- **FR-1.3**: Initialize weights randomly in range [-1, 1] using secure RNG
- **FR-1.4**: Provide deterministic initialization option (for testing)

**Acceptance Criteria:**
```rust
let network = FeedForwardNetwork::new(2, 4, 1); // 2 inputs, 4 hidden, 1 output
assert_eq!(network.layer_count(), 3);
```

### FR-2: Forward Propagation

**Priority:** P0 (Must Have)

- **FR-2.1**: Compute network output given input vector
- **FR-2.2**: Apply linear activation to input/output layers
- **FR-2.3**: Apply sigmoid activation to hidden layer
- **FR-2.4**: Return output vector matching output layer size

**Acceptance Criteria:**
```rust
let output = network.forward(&[0.0, 1.0])?;
assert_eq!(output.len(), 1);
assert!(output[0] >= 0.0 && output[0] <= 1.0);
```

### FR-3: Backpropagation Training

**Priority:** P0 (Must Have)

- **FR-3.1**: Train by fixed iteration count
- **FR-3.2**: Train until error threshold reached
- **FR-3.3**: Update weights using gradient descent (η=0.01 default)
- **FR-3.4**: Calculate deltas for output and hidden layers
- **FR-3.5**: Support batch training (multiple examples per iteration)

**Acceptance Criteria:**
```rust
let inputs = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

network.train_by_error(&inputs, &targets, 0.0001)?;
// Network should learn XOR function
```

### FR-4: Testing & Evaluation

**Priority:** P0 (Must Have)

- **FR-4.1**: Evaluate network on test dataset
- **FR-4.2**: Report accuracy for classification tasks
- **FR-4.3**: Calculate correct/incorrect predictions
- **FR-4.4**: Return classification accuracy percentage

**Acceptance Criteria:**
```rust
let results = network.test(&test_inputs, &test_targets)?;
assert!(results.accuracy >= 90.0);
assert_eq!(results.correct + results.incorrect, test_inputs.len());
```

### FR-5: Data I/O

**Priority:** P0 (Must Have)

- **FR-5.1**: Read matrix data from text files
- **FR-5.2**: Support space-separated float values
- **FR-5.3**: Auto-detect matrix dimensions
- **FR-5.4**: Handle file I/O errors gracefully

**Acceptance Criteria:**
```rust
let matrix = read_matrix_from_file("samples/Xapp.txt")?;
assert_eq!(matrix.len(), 967); // rows
assert_eq!(matrix[0].len(), 55); // columns
```

### FR-6: Network Introspection

**Priority:** P1 (Should Have)

- **FR-6.1**: Query network architecture (layer sizes)
- **FR-6.2**: Access intermediate layer outputs (for debugging)
- **FR-6.3**: Display weight matrices
- **FR-6.4**: Export network statistics

**Acceptance Criteria:**
```rust
println!("{}", network); // Display architecture
let info = network.info();
assert_eq!(info.input_neurons, 2);
```

## Non-Functional Requirements

### NFR-1: Performance

**Priority:** P0

- Training speed within 10% of C++ implementation
- Memory usage proportional to network size
- No unnecessary allocations in hot paths
- Optional BLAS integration for matrix operations

### NFR-2: Reliability

**Priority:** P0

- Zero unsafe code in public API
- Comprehensive error handling (no panics in library code)
- Validated inputs (dimension checking)
- Deterministic behavior (given fixed seed)

### NFR-3: Maintainability

**Priority:** P0

- Idiomatic Rust code (clippy clean)
- Modular architecture (separation of concerns)
- Comprehensive inline documentation
- Clear error messages

### NFR-4: Usability

**Priority:** P0

- Intuitive API following Rust conventions
- Builder pattern for complex configurations
- Helpful error messages with context
- Complete examples for common use cases

### NFR-5: Portability

**Priority:** P1

- Pure Rust implementation (no C bindings required)
- Cross-platform (Linux, macOS, Windows, WASM)
- Minimal dependencies
- Optional features behind feature flags

### NFR-6: Documentation

**Priority:** P0

- README with quickstart guide
- API documentation for all public items
- At least 2 complete examples (XOR, digit recognition)
- Architecture documentation
- Migration guide from C++

## User Stories

### US-1: Train XOR Network
**As a** ML student
**I want to** train a network to learn the XOR function
**So that I** can understand basic neural network concepts

**Acceptance:** Network learns XOR with <0.01 error in <10,000 iterations

### US-2: Digit Recognition
**As a** researcher
**I want to** classify handwritten digits from feature vectors
**So that I** can evaluate network performance on real data

**Acceptance:** Achieves ≥90% accuracy on 967 test examples

### US-3: Custom Architecture
**As a** developer
**I want to** easily change network architecture
**So that I** can experiment with different configurations

**Acceptance:** Change layer sizes via constructor parameters

### US-4: Debug Training
**As a** ML practitioner
**I want to** inspect network state during training
**So that I** can debug convergence issues

**Acceptance:** Access layer outputs, weights, and deltas

## Technical Constraints

### TC-1: Language & Toolchain
- Rust 1.70+ (2021 edition)
- Cargo for build/test/package
- No nightly features in core library

### TC-2: Dependencies
- Maximum 10 direct dependencies
- All dependencies must be:
  - Actively maintained (updated within 6 months)
  - Well-documented
  - Apache/MIT licensed

### TC-3: API Stability
- Follow semantic versioning (SemVer)
- No breaking changes in 0.x without major version bump
- Deprecation warnings before removal

## Out of Scope (v0.1)

The following features are explicitly **not** included in the initial release:

- ❌ Multiple hidden layers (3-layer only for v0.1)
- ❌ Advanced optimizers (SGD with momentum, Adam, etc.)
- ❌ Regularization (L1/L2, dropout)
- ❌ Convolutional/recurrent layers
- ❌ GPU acceleration
- ❌ Mini-batch training
- ❌ Model serialization (save/load)
- ❌ Visualization tools (may be added as separate crate)
- ❌ Python bindings
- ❌ Real-time visualization (SFML equivalent)

These features will be added in future releases as separate example directories.

## Future Roadmap (Post v0.1)

### Project Structure Evolution

As the project grows, it will follow this structure:

```
neural-network-rs/
├── src/               # Core reusable library (Layer, Network, Optimizer, etc.)
├── examples/
│   ├── 01-feedforward/      # v0.1 - Basic FFN (XOR, digit recognition)
│   ├── 02-optimizers/       # v0.2 - SGD, momentum, Adam
│   ├── 03-regularization/   # v0.3 - L1/L2, dropout, early stopping
│   ├── 04-cnn/              # v0.4 - Convolutional networks
│   ├── 05-rnn/              # v0.5 - Recurrent networks (LSTM, GRU)
│   ├── 06-gan/              # v0.6 - Generative adversarial networks
│   ├── 07-transformers/     # v0.7 - Attention mechanisms
│   └── ...                  # More as techniques evolve
└── docs/
    ├── tutorials/           # Step-by-step learning guides
    └── papers/              # Reference implementations
```

Each `examples/XX-topic/` directory will contain:
- Complete working code
- README with theory and concepts
- Visualizations of results
- Performance benchmarks
- References to papers/resources

### Guiding Principles

1. **Incremental Learning**: Each example builds on previous concepts
2. **Reusable Core**: All examples use shared library components
3. **Educational Focus**: Code is clear, well-documented, tutorial-oriented
4. **Production Quality**: Performance and correctness remain priorities
5. **Community Driven**: Accept contributions for new techniques/examples

## Example Usage

### Basic XOR Example

```rust
use neural_network_rs::FeedForwardNetwork;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create network: 2 inputs, 4 hidden neurons, 1 output
    let mut network = FeedForwardNetwork::new(2, 4, 1);

    // Training data for XOR
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    // Train until error < 0.0001
    network.train_by_error(&inputs, &targets, 0.0001)?;

    // Test the network
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let output = network.forward(input)?;
        println!("XOR({}, {}) = {:.4} (expected {:.4})",
                 input[0], input[1], output[0], target[0]);
    }

    Ok(())
}
```

### Digit Recognition Example

```rust
use neural_network_rs::{FeedForwardNetwork, utils::read_matrix_from_file};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create network: 55 inputs, 20 hidden, 10 outputs (digits 0-9)
    let mut network = FeedForwardNetwork::new(55, 20, 10);

    // Load pre-processed data
    let train_x = read_matrix_from_file("samples/Xapp.txt")?;
    let train_y = read_matrix_from_file("samples/TA.txt")?;
    let test_x = read_matrix_from_file("samples/Xtest.txt")?;
    let test_y = read_matrix_from_file("samples/TT.txt")?;

    // Train for 1000 iterations
    network.train_by_iteration(&train_x, &train_y, 1000)?;

    // Evaluate on test set
    let results = network.test(&test_x, &test_y)?;

    println!("Correct: {}", results.correct);
    println!("Incorrect: {}", results.incorrect);
    println!("Accuracy: {:.2}%", results.accuracy);

    Ok(())
}
```

## Milestones & Deliverables

### Milestone 1: Core Implementation (Week 1-2)
- [ ] Project structure & build system
- [ ] Layer implementation
- [ ] Network implementation
- [ ] Forward propagation
- [ ] Backpropagation
- [ ] Basic unit tests

### Milestone 2: Training & Testing (Week 3)
- [ ] Training algorithms (by iteration, by error)
- [ ] Testing/evaluation utilities
- [ ] XOR example working
- [ ] Integration tests

### Milestone 3: Data I/O (Week 4)
- [ ] File reading utilities
- [ ] Digit recognition example working
- [ ] Error handling improvements
- [ ] Documentation

### Milestone 4: Polish & Release (Week 5)
- [ ] API documentation complete
- [ ] README with examples
- [ ] Architecture docs
- [ ] Benchmarks vs C++
- [ ] Publish to crates.io (optional)

## Dependencies & Risks

### Dependencies
- `ndarray` - mature, widely used, stable API
- `rand` - standard library for Rust, stable
- `thiserror` - popular error handling, stable

### Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Performance slower than C++ | High | Medium | Profile and optimize; use BLAS |
| API design needs changes | Medium | Low | Follow Rust conventions; review early |
| Integration tests fail | High | Low | Port C++ tests directly |
| Missing edge cases | Medium | Medium | Comprehensive test suite |

## Appendix

### References
- Original C++ Repository: [Feed-Forward-Neural-Network](https://github.com/wimacod/Feed-Forward-Neural-Network)
- Backpropagation Tutorial: http://www.cs.bham.ac.uk/~jxb/NN/l7.pdf
- FNN Theory: http://www.di.unito.it/~cancelli/retineu06_07/FNN.pdf

### Glossary
- **FFN**: Feed-Forward Network
- **Backpropagation**: Algorithm for training neural networks via gradient descent
- **Epoch**: One complete pass through training data
- **Sigmoid**: S-shaped activation function: f(x) = 1/(1+e^-x)
- **Delta**: Error gradient for a neuron
- **Eta (η)**: Learning rate hyperparameter
