# Neural Network Examples in Rust

A comprehensive machine learning demonstration platform built in Rust, showcasing neural network architectures and techniques through incremental, educational examples.

## Vision

This project aims to create an **educational ML platform** with:

- **Reusable Core Library:** Type-safe, high-performance implementations of fundamental ML components
- **Incremental Examples:** Step-by-step progression from basic to advanced concepts
- **Comprehensive Tutorials:** Each example includes theory, code, visualizations, and benchmarks
- **Production Quality:** Leveraging Rust's safety guarantees and zero-cost abstractions

## Current Status: v0.1 - Foundation

**Phase 0 Complete:** Project structure, core data types, and error handling ✓

**Next:** Implementing forward propagation (Phase 2)

### What's Implemented

- ✓ Core error types (`NeuralNetError`, `Result`)
- ✓ Activation functions (Sigmoid, Linear) with trait-based design
- ✓ Layer structure with weight initialization
- ✓ Network structure (3-layer FFN skeleton)
- ✓ File I/O utilities for matrix data
- ✓ Comprehensive test suite (13 unit tests passing)

### In Progress

- Forward propagation algorithm
- Backpropagation and training
- XOR learning example
- Digit recognition example

## Quick Start

### Prerequisites

- Rust 1.70+ ([Install Rust](https://www.rust-lang.org/tools/install))
- Cargo (included with Rust)

### Build & Test

```bash
# Clone the repository
git clone https://github.com/softwarewrighter/neural-network-examples-rs.git
cd neural-network-examples-rs

# Build the project
cargo build

# Run tests
cargo test

# Run linting
cargo clippy -- -D warnings

# Generate documentation
cargo doc --open
```

### Usage (When Complete)

```rust
use neural_network_rs::FeedForwardNetwork;

// Create a network: 2 inputs, 4 hidden neurons, 1 output
let mut network = FeedForwardNetwork::new(2, 4, 1);

// Training data for XOR
let inputs = vec![
    vec![0.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 0.0],
    vec![1.0, 1.0],
];
let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

// Train the network
network.train_by_error(&inputs, &targets, 0.0001)?;

// Test the network
let output = network.forward(&[1.0, 0.0])?;
println!("XOR(1, 0) = {:.4}", output[0]); // Expected: ~1.0
```

## Project Structure

```
neural-network-rs/
├── src/                    # Core library (reusable components)
│   ├── lib.rs             # Public API
│   ├── error.rs           # Error types
│   ├── activation.rs      # Activation functions
│   ├── layer.rs           # Layer implementation
│   ├── network.rs         # Network implementation
│   └── utils/             # Utilities (file I/O, etc.)
├── examples/              # Example programs (XOR, digit recognition)
├── tests/                 # Integration tests
├── benches/               # Performance benchmarks
├── docs/                  # Documentation (see below)
└── samples/               # Training/test data
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[docs/architecture.md](docs/architecture.md)** - Technical architecture, design decisions, and patterns
- **[docs/PRD.md](docs/PRD.md)** - Product requirements, goals, success metrics, and roadmap
- **[docs/plan.md](docs/plan.md)** - Detailed 6-phase implementation plan with tasks and timelines
- **[docs/learnings.md](docs/learnings.md)** - Key decisions, rationale, and lessons learned

**Start here:** Read `docs/PRD.md` for project goals, then `docs/architecture.md` for technical details.

## Roadmap

### v0.1 - Feed-Forward Network Foundation (Current)
- ✓ Project setup and core data structures
- ⏳ Forward propagation
- ⏳ Backpropagation and training
- ⏳ XOR learning example
- ⏳ Digit recognition example

### v0.2+ - Incremental ML Techniques (Future)

The project will evolve into a structured learning platform:

```
examples/
├── 01-feedforward/      # Basic FFN (v0.1)
├── 02-optimizers/       # SGD, Momentum, Adam (v0.2)
├── 03-regularization/   # L1/L2, Dropout (v0.3)
├── 04-cnn/              # Convolutional networks (v0.4)
├── 05-rnn/              # Recurrent networks (v0.5)
├── 06-gan/              # Generative models (v0.6)
└── ...                  # More as techniques evolve
```

Each example directory will include:
- README with theory and concepts
- Working code with extensive comments
- Visualizations of results
- Performance benchmarks
- References to papers/resources

## Development Approach

### Test-Driven Development (TDD)

We use Red-Green-Refactor methodology:
1. **Red:** Write failing test defining desired behavior
2. **Green:** Implement minimal code to pass test
3. **Refactor:** Improve code quality while keeping tests green

### Local CI

Quality checks run on local infrastructure:
```bash
cargo test                     # All tests must pass
cargo clippy -- -D warnings    # Zero warnings required
cargo fmt -- --check           # Consistent formatting
cargo doc --no-deps            # Documentation builds
```

## Dependencies

Core dependencies:
- `ndarray` (0.15) - Multi-dimensional arrays with BLAS integration
- `rand` (0.8) - Random number generation
- `thiserror` (1.0) - Ergonomic error types

Development dependencies:
- `approx` (0.5) - Float comparison in tests
- `criterion` (0.5) - Statistical benchmarking

## Contributing

This is currently an educational project in active development. Contributions, suggestions, and discussions are welcome!

### Areas for Future Contribution
- Additional activation functions (ReLU, Tanh, etc.)
- New network architectures
- Optimization algorithms
- Visualization tools
- Tutorial documentation
- Performance optimizations

## License

MIT License (see [LICENSE](LICENSE))

## References

### Original C++ Implementation
- [Feed-Forward-Neural-Network](https://github.com/wimacod/Feed-Forward-Neural-Network)
- See `research/README-cpp-original.md` for original project documentation

### Neural Network Theory
- [Backpropagation Tutorial](http://www.cs.bham.ac.uk/~jxb/NN/l7.pdf)
- [Feed-Forward Neural Networks](http://www.di.unito.it/~cancelli/retineu06_07/FNN.pdf)

### Rust ML Ecosystem
- [Are We Learning Yet?](http://www.arewelearningyet.com/) - Rust ML crate directory
- [ndarray documentation](https://docs.rs/ndarray/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)

## Acknowledgments

- Original C++ implementation by Alexis Louis
- Inspired by classic backpropagation algorithms and educational ML resources
- Built with Rust's powerful type system and zero-cost abstractions

---

**Status:** Phase 0 complete, Phase 2 (Forward Propagation) ready to begin

**Last Updated:** 2025-10-14
