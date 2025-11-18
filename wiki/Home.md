# Neural Network Examples - Rust

Welcome to the **Neural Network Examples in Rust** wiki! This educational ML platform demonstrates neural network concepts through clean, well-tested Rust implementations.

## Project Overview

This project is an **educational ML demonstration platform** built in Rust, featuring:

- **Reusable core library** with type-safe ML components organized into focused crates
- **Incremental examples** showcasing neural network concepts (XOR, boolean logic, etc.)
- **Interactive visualizations** through a Yew-based WASM frontend
- **Comprehensive documentation** with architecture diagrams and tutorials

**Current Status:** Phase 2 complete - Yew WASM frontend with interactive network visualization

## Quick Navigation

### Architecture & Design
- [[Architecture-Overview]] - System architecture with component diagrams
- [[Core-Components]] - Detailed crate structure and responsibilities
- [[Data-Flow]] - Sequence diagrams showing data flow through the system

### Implementation Details
- [[Training-Algorithms]] - Forward/backward propagation and optimization algorithms
- [[Activation-Functions]] - Sigmoid, Linear, and modern activation implementations
- [[Error-Handling]] - Result-based error handling strategy

### Examples & Tutorials
- [[Example-Structure]] - How examples are organized and tested
- [[XOR-Example]] - Classic XOR problem walkthrough
- [[Boolean-Logic-Examples]] - AND, OR, Majority functions

### Developer Resources
- [[Development-Process]] - TDD workflow, code quality standards
- [[Testing-Strategy]] - Unit, integration, and negative testing patterns
- [[Contributing]] - How to contribute to the project

## Key Features

### 🎯 Clean Architecture
- **Separation of concerns**: Types, algorithms, visualization in separate crates
- **No circular dependencies**: Clear dependency graph
- **Small, focused crates**: Each <500 LOC for maintainability

### 🔒 Type Safety
- **Ownership model**: Network owns layers (unidirectional)
- **No unsafe code**: Leverages Rust's safety guarantees
- **Result-based errors**: Comprehensive error handling

### 📊 Interactive Visualization
- **Live network visualization**: See weights, neurons, and data flow
- **WASM frontend**: Runs in browser via Yew framework
- **SVG generation**: Export network diagrams

### 🧪 Test-Driven
- **TDD workflow**: Red-Green-Refactor for all features
- **Comprehensive tests**: Negative + positive tests for all examples
- **Zero warnings**: Clippy compliance enforced

## Project Structure

```
neural-network-examples-rs/
├── crates/                    # Library crates
│   ├── neural-net-types/      # Data structures & serialization
│   ├── neural-net-core/       # Algorithms & computation
│   ├── neural-net-viz/        # SVG visualization
│   └── neural-net-animator/   # Animation framework
├── examples/                  # Educational examples
│   ├── example-1-forward-propagation-xor/
│   ├── example-2-backward-propagation-xor/
│   └── ...
├── yew-app/                   # WASM frontend
└── documentation/             # Architecture & planning docs
```

See [[Core-Components]] for detailed crate descriptions.

## Getting Started

### Prerequisites
- Rust 1.85+ (2024 edition)
- Cargo for building and testing

### Build & Test
```bash
# Build all crates
cargo build

# Run tests
cargo test

# Run an example
cargo run --example example-2-backward-propagation-xor

# Serve the Yew frontend
cd yew-app && trunk serve
```

### Quick Example: XOR Network

```rust
use neural_net_core::prelude::*;
use neural_net_types::{FeedForwardNetwork, Layer};

// Create network: 2 inputs, 2 hidden, 1 output
let mut network = FeedForwardNetwork::new_with_config(2, 2, 1)?;

// XOR training data
let inputs = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
let targets = vec![[0.0], [1.0], [1.0], [0.0]];

// Train the network
for _ in 0..10000 {
    for (input, target) in inputs.iter().zip(targets.iter()) {
        network.train_single(input, target, 0.5)?;
    }
}

// Test the network
for (input, expected) in inputs.iter().zip(targets.iter()) {
    let output = network.forward(input)?;
    println!("XOR({}, {}) = {:.4}", input[0], input[1], output[0]);
}
```

## Documentation Resources

### In Repository
- [Architecture Document](../../blob/main/documentation/architecture.md) - Comprehensive architecture guide
- [PRD](../../blob/main/documentation/PRD.md) - Product requirements and roadmap
- [Process Guide](../../blob/main/documentation/process.md) - Development workflow (TDD, quality checks)
- [Learnings](../../blob/main/documentation/learnings.md) - Lessons learned and patterns

### Live Demo
- [GitHub Pages Demo](https://softwarewrighter.github.io/neural-network-examples-rs/) - Interactive WASM frontend

## Core Design Principles

1. **Architecture over features**: Clean, well-tested code takes precedence
2. **Educational clarity**: Code should be easy to understand and learn from
3. **Type safety first**: Leverage Rust's type system for correctness
4. **Test-driven development**: Red-Green-Refactor for all features
5. **Zero warnings**: Clippy compliance is mandatory

## Technology Stack

- **Language**: Rust 2024 edition
- **Frontend**: Yew (WASM)
- **Math**: ndarray for matrix operations
- **Testing**: Built-in Rust testing + approx for float comparison
- **Visualization**: SVG generation

## Contributing

Contributions are welcome! Please see our [[Contributing]] guide for:
- Code style guidelines
- Testing requirements
- Pull request process
- Development workflow

## Version History

- **Phase 0** ✅: Foundation - types, errors, activation traits
- **Phase 1** ✅: Forward propagation and infrastructure
- **Phase 2** ✅: Yew WASM frontend with network visualization
- **Phase 3** 🚧: Advanced features and additional examples

## External References

- [Original C++ Implementation](https://github.com/wimacod/Feed-Forward-Neural-Network)
- [Backpropagation Tutorial](http://www.cs.bham.ac.uk/~jxb/NN/l7.pdf)
- [FNN Theory](http://www.di.unito.it/~cancelli/retineu06_07/FNN.pdf)

## License

This project is open source. See [LICENSE](../../blob/main/LICENSE) for details.

---

**Next Steps:**
- Explore the [[Architecture-Overview]] to understand system design
- Read about [[Core-Components]] to learn the crate structure
- Check out [[Training-Algorithms]] for implementation details
- Try running the examples in the repository
