# Implementation Plan - C++ to Rust Migration

## Overview

This document outlines the step-by-step plan for porting the C++ feed-forward neural network to Rust. The plan is organized into phases, each with specific tasks, estimated effort, and success criteria.

**Total Estimated Time:** 4-5 weeks
**Team Size:** 1-2 developers
**Approach:** Incremental migration with continuous testing

## Phase 0: Project Setup (2-3 days)

### Goals
- Initialize Rust project structure
- Set up development environment
- Configure tooling and CI

### Tasks

#### 0.1: Initialize Cargo Project
```bash
cargo new --lib neural-network-rs
cd neural-network-rs
```

**Deliverables:**
- `Cargo.toml` configured with metadata
- Initial directory structure
- `.gitignore` for Rust

#### 0.2: Configure Dependencies

Edit `Cargo.toml`:
```toml
[package]
name = "neural-network-rs"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <email@example.com>"]
license = "MIT"
description = "Feed-forward neural network with backpropagation"
repository = "https://github.com/username/neural-network-rs"

[dependencies]
ndarray = "0.15"
rand = "0.8"
thiserror = "1.0"

[dev-dependencies]
approx = "0.5"
criterion = "0.5"

[[bench]]
name = "training_benchmark"
harness = false
```

#### 0.3: Set Up Directory Structure

```
neural-network-rs/
├── Cargo.toml
├── README.md
├── LICENSE
├── src/
│   ├── lib.rs
│   ├── network.rs          # (to be created)
│   ├── layer.rs            # (to be created)
│   ├── activation.rs       # (to be created)
│   ├── trainer.rs          # (to be created)
│   ├── testing.rs          # (to be created)
│   └── utils/
│       ├── mod.rs
│       └── file_io.rs      # (to be created)
├── examples/
│   ├── xor.rs              # (to be created)
│   └── digit_recognition.rs # (to be created)
├── tests/
│   └── integration_tests.rs # (to be created)
├── benches/
│   └── training_benchmark.rs # (to be created)
├── samples/                 # Copy from C++ project
│   ├── Xapp.txt
│   ├── TA.txt
│   ├── Xtest.txt
│   └── TT.txt
└── docs/
    ├── architecture.md      # ✓ Done
    ├── PRD.md              # ✓ Done
    └── plan.md             # ✓ Done
```

#### 0.4: Copy Sample Data
```bash
cp -r ../Feed-Forward-Neural-Network/samples ./
```

#### 0.5: Configure CI (Optional but Recommended)

Create `.github/workflows/ci.yml`:
```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --all-features
      - run: cargo clippy -- -D warnings
      - run: cargo fmt -- --check
```

**Success Criteria:**
- [ ] `cargo build` succeeds
- [ ] `cargo test` runs (no tests yet, but command works)
- [ ] Directory structure matches plan
- [ ] Sample data files present

---

## Phase 1: Core Data Structures (3-4 days)

### Goals
- Implement Layer struct
- Implement FeedForwardNetwork struct
- Set up error handling
- Write basic unit tests

### Tasks

#### 1.1: Create Error Types

**File:** `src/lib.rs`

```rust
use thiserror::Error;

#[derive(Debug, Error)]
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

pub type Result<T> = std::result::Result<T, NeuralNetError>;
```

**Reference:** No direct C++ equivalent (uses exceptions/unchecked errors)

**Test:**
```rust
#[test]
fn test_error_display() {
    let err = NeuralNetError::DimensionMismatch { expected: 3, actual: 5 };
    assert!(err.to_string().contains("expected 3"));
}
```

#### 1.2: Implement Activation Functions

**File:** `src/activation.rs`

```rust
pub trait Activation {
    fn activate(&self, x: f32) -> f32;
    fn derivative(&self, output: f32) -> f32;
}

#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    #[inline]
    fn activate(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[inline]
    fn derivative(&self, output: f32) -> f32 {
        output * (1.0 - output)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Linear;

impl Activation for Linear {
    #[inline]
    fn activate(&self, x: f32) -> f32 {
        x
    }

    #[inline]
    fn derivative(&self, _output: f32) -> f32 {
        1.0
    }
}
```

**Reference:** `Layer.cpp:62` (sigmoid), `Layer.cpp:58-59` (linear)

**Tests:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sigmoid() {
        let sigmoid = Sigmoid;
        assert_relative_eq!(sigmoid.activate(0.0), 0.5);
        assert_relative_eq!(sigmoid.activate(10.0), 0.9999, epsilon = 0.0001);
        assert_relative_eq!(sigmoid.derivative(0.5), 0.25);
    }

    #[test]
    fn test_linear() {
        let linear = Linear;
        assert_eq!(linear.activate(5.0), 5.0);
        assert_eq!(linear.derivative(10.0), 1.0);
    }
}
```

#### 1.3: Implement Layer Struct

**File:** `src/layer.rs`

Port from `Layer.hpp` and `Layer.cpp`. Key mappings:

| C++ | Rust |
|-----|------|
| `int indice` | `index: usize` |
| `int nb_neurons` | `num_neurons: usize` |
| `vector<vector<float>> weights` | `weights: Option<Array2<f32>>` |
| `vector<float> inputs` | `inputs: Vec<f32>` |
| `vector<float> outputs` | `outputs: Vec<f32>` |
| `vector<float> deltas` | `deltas: Vec<f32>` |
| `FFN *network` | Removed (use references in methods) |

**Implementation outline:**
```rust
use ndarray::Array2;
use rand::Rng;

pub struct Layer {
    index: usize,
    num_neurons: usize,
    weights: Option<Array2<f32>>,
    inputs: Vec<f32>,
    outputs: Vec<f32>,
    deltas: Vec<f32>,
}

impl Layer {
    pub fn new(index: usize, num_neurons: usize, prev_layer_size: Option<usize>) -> Self {
        let weights = prev_layer_size.map(|prev_size| {
            let mut rng = rand::thread_rng();
            Array2::from_shape_fn((prev_size, num_neurons), |_| {
                rng.gen_range(-1.0..1.0)
            })
        });

        Self {
            index,
            num_neurons,
            weights,
            inputs: Vec::with_capacity(num_neurons),
            outputs: Vec::with_capacity(num_neurons),
            deltas: Vec::with_capacity(num_neurons),
        }
    }

    // Getters
    pub fn outputs(&self) -> &[f32] { &self.outputs }
    pub fn inputs(&self) -> &[f32] { &self.inputs }
    pub fn deltas(&self) -> &[f32] { &self.deltas }
    pub fn weights(&self) -> Option<&Array2<f32>> { self.weights.as_ref() }

    // Setters
    pub fn set_inputs(&mut self, inputs: Vec<f32>) {
        self.inputs = inputs;
    }

    // To be implemented in Phase 2
    pub fn forward_propagate(&mut self, prev_outputs: Option<&[f32]>) {
        todo!("Phase 2")
    }

    pub fn calc_deltas(&mut self, /* params */) {
        todo!("Phase 2")
    }

    pub fn update_weights(&mut self, /* params */) {
        todo!("Phase 2")
    }
}
```

**Reference:** `Layer.cpp:11-29` (constructor)

**Tests:**
```rust
#[test]
fn test_layer_creation() {
    let layer = Layer::new(0, 3, None); // Input layer
    assert_eq!(layer.num_neurons, 3);
    assert!(layer.weights.is_none());

    let hidden = Layer::new(1, 4, Some(3));
    assert_eq!(hidden.num_neurons, 4);
    assert_eq!(hidden.weights.as_ref().unwrap().shape(), &[3, 4]);
}

#[test]
fn test_weight_range() {
    let layer = Layer::new(1, 10, Some(5));
    let weights = layer.weights.unwrap();
    for &w in weights.iter() {
        assert!(w >= -1.0 && w <= 1.0);
    }
}
```

#### 1.4: Implement Network Struct (Skeleton)

**File:** `src/network.rs`

```rust
use crate::layer::Layer;
use crate::Result;

pub struct FeedForwardNetwork {
    layers: Vec<Layer>,
    targets: Option<Vec<f32>>,
}

impl FeedForwardNetwork {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut layers = Vec::with_capacity(3);

        // Input layer (no weights)
        layers.push(Layer::new(0, input_size, None));

        // Hidden layer
        layers.push(Layer::new(1, hidden_size, Some(input_size)));

        // Output layer
        layers.push(Layer::new(2, output_size, Some(hidden_size)));

        Self {
            layers,
            targets: None,
        }
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer(&self, index: usize) -> Option<&Layer> {
        self.layers.get(index)
    }

    pub fn layer_mut(&mut self, index: usize) -> Option<&mut Layer> {
        self.layers.get_mut(index)
    }

    // To be implemented in Phase 2 & 3
    pub fn forward(&mut self, inputs: &[f32]) -> Result<Vec<f32>> {
        todo!("Phase 2")
    }

    pub fn train_by_iteration(/* ... */) -> Result<()> {
        todo!("Phase 3")
    }

    pub fn train_by_error(/* ... */) -> Result<()> {
        todo!("Phase 3")
    }

    pub fn test(/* ... */) -> Result<TestResults> {
        todo!("Phase 3")
    }
}
```

**Reference:** `FFN.cpp:18-22` (initialization)

**Tests:**
```rust
#[test]
fn test_network_creation() {
    let net = FeedForwardNetwork::new(2, 4, 1);
    assert_eq!(net.layer_count(), 3);
    assert_eq!(net.layer(0).unwrap().num_neurons, 2);
    assert_eq!(net.layer(1).unwrap().num_neurons, 4);
    assert_eq!(net.layer(2).unwrap().num_neurons, 1);
}
```

#### 1.5: Update lib.rs

**File:** `src/lib.rs`

```rust
mod activation;
mod layer;
mod network;

pub use activation::{Activation, Linear, Sigmoid};
pub use network::FeedForwardNetwork;

// Error types
mod error;
pub use error::{NeuralNetError, Result};
```

**Success Criteria:**
- [ ] All Phase 1 code compiles without warnings
- [ ] `cargo test` passes for implemented tests
- [ ] `cargo clippy` reports no issues
- [ ] Basic network can be constructed

---

## Phase 2: Forward Propagation (2-3 days)

### Goals
- Implement forward pass algorithm
- Test with simple inputs
- Verify output ranges

### Tasks

#### 2.1: Implement Layer::calc_inputs()

**File:** `src/layer.rs`

Port from `Layer.cpp:41-54`:

```rust
impl Layer {
    fn calc_inputs(&mut self, prev_outputs: &[f32]) -> Result<()> {
        if self.index == 0 {
            return Ok(()); // Input layer has no calculation
        }

        let weights = self.weights.as_ref()
            .ok_or_else(|| NeuralNetError::InvalidConfig("No weights for non-input layer".into()))?;

        if weights.shape()[0] != prev_outputs.len() {
            return Err(NeuralNetError::DimensionMismatch {
                expected: weights.shape()[0],
                actual: prev_outputs.len(),
            });
        }

        self.inputs.clear();

        // Matrix-vector multiplication: inputs = weights^T * prev_outputs
        for col in 0..weights.shape()[1] {
            let mut sum = 0.0;
            for row in 0..weights.shape()[0] {
                sum += prev_outputs[row] * weights[[row, col]];
            }
            self.inputs.push(sum);
        }

        Ok(())
    }
}
```

**Reference:** `Layer.cpp:41-54`

**Test:**
```rust
#[test]
fn test_calc_inputs() {
    let mut layer = Layer::new(1, 2, Some(3));
    // Set known weights for testing
    layer.weights = Some(Array2::from_shape_vec((3, 2), vec![
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    ]).unwrap());

    let prev_outputs = vec![1.0, 2.0, 3.0];
    layer.calc_inputs(&prev_outputs).unwrap();

    // Expected: [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6] = [22, 28]
    assert_eq!(layer.inputs, vec![22.0, 28.0]);
}
```

#### 2.2: Implement Layer::calc_outputs()

**File:** `src/layer.rs`

Port from `Layer.cpp:56-65`:

```rust
use crate::activation::{Activation, Linear, Sigmoid};

impl Layer {
    fn calc_outputs(&mut self, is_output_layer: bool) {
        self.outputs.clear();

        if self.index == 0 || is_output_layer {
            // Linear activation for input/output layers
            self.outputs.extend_from_slice(&self.inputs);
        } else {
            // Sigmoid activation for hidden layers
            let sigmoid = Sigmoid;
            for &input in &self.inputs {
                self.outputs.push(sigmoid.activate(input));
            }
        }
    }
}
```

**Reference:** `Layer.cpp:56-65`

**Test:**
```rust
#[test]
fn test_calc_outputs_linear() {
    let mut layer = Layer::new(0, 3, None);
    layer.inputs = vec![1.0, 2.0, 3.0];
    layer.calc_outputs(false);
    assert_eq!(layer.outputs, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_calc_outputs_sigmoid() {
    let mut layer = Layer::new(1, 2, Some(3));
    layer.inputs = vec![0.0, 10.0];
    layer.calc_outputs(false);

    assert_relative_eq!(layer.outputs[0], 0.5);
    assert!(layer.outputs[1] > 0.999);
}
```

#### 2.3: Implement Layer::forward_propagate()

**File:** `src/layer.rs`

Port from `Layer.cpp:67-70`:

```rust
impl Layer {
    pub fn forward_propagate(
        &mut self,
        prev_outputs: Option<&[f32]>,
        is_output_layer: bool
    ) -> Result<()> {
        if let Some(prev) = prev_outputs {
            self.calc_inputs(prev)?;
        }
        self.calc_outputs(is_output_layer);
        Ok(())
    }
}
```

**Reference:** `Layer.cpp:67-70`

#### 2.4: Implement Network::forward()

**File:** `src/network.rs`

Port from `FFN.cpp:24-29`:

```rust
impl FeedForwardNetwork {
    pub fn forward(&mut self, inputs: &[f32]) -> Result<Vec<f32>> {
        // Validate input size
        if inputs.len() != self.layers[0].num_neurons {
            return Err(NeuralNetError::DimensionMismatch {
                expected: self.layers[0].num_neurons,
                actual: inputs.len(),
            });
        }

        // Set input layer
        self.layers[0].set_inputs(inputs.to_vec());
        self.layers[0].forward_propagate(None, false)?;

        // Propagate through hidden and output layers
        for i in 1..self.layers.len() {
            let is_output = i == self.layers.len() - 1;
            let prev_outputs = self.layers[i - 1].outputs().to_vec();
            self.layers[i].forward_propagate(Some(&prev_outputs), is_output)?;
        }

        // Return output layer's outputs
        Ok(self.layers.last().unwrap().outputs().to_vec())
    }
}
```

**Reference:** `FFN.cpp:24-29`

**Tests:**
```rust
#[test]
fn test_forward_pass() {
    let mut net = FeedForwardNetwork::new(2, 3, 1);
    let output = net.forward(&[0.5, 0.5]).unwrap();

    assert_eq!(output.len(), 1);
    // Output should be in valid range (no specific value yet)
}

#[test]
fn test_forward_dimension_mismatch() {
    let mut net = FeedForwardNetwork::new(2, 3, 1);
    let result = net.forward(&[0.5]); // Wrong size

    assert!(matches!(result, Err(NeuralNetError::DimensionMismatch { .. })));
}
```

**Success Criteria:**
- [ ] Forward pass executes without errors
- [ ] Output dimensions match network configuration
- [ ] Hidden layer uses sigmoid activation
- [ ] Input/output layers use linear activation
- [ ] Dimension mismatches are caught and reported

---

## Phase 3: Backpropagation (3-4 days)

### Goals
- Implement backpropagation algorithm
- Implement weight updates
- Test with XOR problem

### Tasks

#### 3.1: Implement Layer::calc_deltas()

**File:** `src/layer.rs`

Port from `Layer.cpp:77-98`:

```rust
impl Layer {
    pub fn calc_deltas(
        &mut self,
        targets: Option<&[f32]>,
        next_layer: Option<&Layer>,
    ) -> Result<()> {
        if self.index == 0 {
            return Ok(()); // Input layer has no deltas
        }

        self.deltas.clear();

        if let Some(targets) = targets {
            // Output layer: δ = (target - output)
            for (i, &target) in targets.iter().enumerate() {
                self.deltas.push(target - self.outputs[i]);
            }
        } else if let Some(next) = next_layer {
            // Hidden layer: δ = Σ(next_weights * next_deltas) * output * (1 - output)
            let next_weights = next.weights.as_ref().unwrap();
            let next_deltas = next.deltas();

            for i in 0..self.num_neurons {
                let mut delta_sum = 0.0;
                for j in 0..next_deltas.len() {
                    delta_sum += next_weights[[i, j]] * next_deltas[j];
                }
                // Derivative of sigmoid: output * (1 - output)
                delta_sum *= self.outputs[i] * (1.0 - self.outputs[i]);
                self.deltas.push(delta_sum);
            }
        }

        Ok(())
    }
}
```

**Reference:** `Layer.cpp:77-98`

**Test:**
```rust
#[test]
fn test_output_layer_deltas() {
    let mut layer = Layer::new(2, 2, Some(3));
    layer.outputs = vec![0.8, 0.3];

    let targets = vec![1.0, 0.0];
    layer.calc_deltas(Some(&targets), None).unwrap();

    assert_relative_eq!(layer.deltas[0], 0.2);  // 1.0 - 0.8
    assert_relative_eq!(layer.deltas[1], -0.3); // 0.0 - 0.3
}
```

#### 3.2: Implement Layer::update_weights()

**File:** `src/layer.rs`

Port from `Layer.cpp:100-110`:

```rust
impl Layer {
    pub fn update_weights(&mut self, prev_outputs: &[f32], learning_rate: f32) -> Result<()> {
        if self.index == 0 {
            return Ok(()); // Input layer has no weights
        }

        let weights = self.weights.as_mut()
            .ok_or_else(|| NeuralNetError::InvalidConfig("No weights to update".into()))?;

        for i in 0..weights.shape()[0] {
            for j in 0..weights.shape()[1] {
                weights[[i, j]] += learning_rate * self.deltas[j] * prev_outputs[i];
            }
        }

        Ok(())
    }
}
```

**Reference:** `Layer.cpp:100-110`

**Test:**
```rust
#[test]
fn test_weight_update() {
    let mut layer = Layer::new(1, 2, Some(2));
    layer.deltas = vec![0.1, -0.2];

    let initial_weights = layer.weights.as_ref().unwrap().clone();
    let prev_outputs = vec![0.5, 0.5];

    layer.update_weights(&prev_outputs, 0.01).unwrap();

    let updated_weights = layer.weights.as_ref().unwrap();

    // Verify weights changed
    assert_ne!(initial_weights, *updated_weights);
}
```

#### 3.3: Implement Training Methods

**File:** `src/network.rs`

Port from `FFN.cpp:84-109` (train_by_iteration) and `FFN.cpp:57-82` (train_by_error):

```rust
impl FeedForwardNetwork {
    pub fn train_by_iteration(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        iterations: usize,
    ) -> Result<()> {
        const LEARNING_RATE: f32 = 0.01;

        for iteration in 0..iterations {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                // Forward pass
                self.forward(input)?;

                // Set targets
                self.targets = Some(target.clone());

                // Backward pass
                for i in (1..self.layers.len()).rev() {
                    let is_output = i == self.layers.len() - 1;

                    if is_output {
                        let targets = self.targets.as_ref().unwrap();
                        self.layers[i].calc_deltas(Some(targets), None)?;
                    } else {
                        // Need next layer for delta calculation
                        let (left, right) = self.layers.split_at_mut(i + 1);
                        let next_layer = &right[0];
                        left[i].calc_deltas(None, Some(next_layer))?;
                    }

                    // Update weights
                    let prev_outputs = self.layers[i - 1].outputs().to_vec();
                    self.layers[i].update_weights(&prev_outputs, LEARNING_RATE)?;
                }
            }
        }

        println!("Stopping at iteration: {}", iterations);
        Ok(())
    }

    pub fn train_by_error(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        target_error: f32,
    ) -> Result<()> {
        const LEARNING_RATE: f32 = 0.01;
        let mut error = f32::MAX;

        while error > target_error {
            error = 0.0;

            for (input, target) in inputs.iter().zip(targets.iter()) {
                // Forward pass
                let outputs = self.forward(input)?;

                // Calculate error (MSE)
                for (i, &t) in target.iter().enumerate() {
                    let diff = t - outputs[i];
                    error += diff * diff;
                }

                // Backward pass (same as train_by_iteration)
                self.targets = Some(target.clone());

                for i in (1..self.layers.len()).rev() {
                    let is_output = i == self.layers.len() - 1;

                    if is_output {
                        let targets = self.targets.as_ref().unwrap();
                        self.layers[i].calc_deltas(Some(targets), None)?;
                    } else {
                        let (left, right) = self.layers.split_at_mut(i + 1);
                        let next_layer = &right[0];
                        left[i].calc_deltas(None, Some(next_layer))?;
                    }

                    let prev_outputs = self.layers[i - 1].outputs().to_vec();
                    self.layers[i].update_weights(&prev_outputs, LEARNING_RATE)?;
                }
            }
        }

        println!("Stopping at error: {}", error);
        Ok(())
    }
}
```

**Reference:** `FFN.cpp:57-109`

**Success Criteria:**
- [ ] Training loop executes without errors
- [ ] Weights are updated after each iteration
- [ ] Error decreases over time (for train_by_error)

---

## Phase 4: Testing & Utilities (2-3 days)

### Goals
- Implement test/evaluation functionality
- Implement file I/O utilities
- Create example programs

### Tasks

#### 4.1: Implement Network::test()

**File:** `src/network.rs` or `src/testing.rs`

Port from `FFN.cpp:31-55`:

```rust
#[derive(Debug)]
pub struct TestResults {
    pub correct: usize,
    pub incorrect: usize,
    pub accuracy: f32,
}

impl FeedForwardNetwork {
    pub fn test(
        &mut self,
        test_inputs: &[Vec<f32>],
        test_targets: &[Vec<f32>],
    ) -> Result<TestResults> {
        let mut correct = 0;
        let mut incorrect = 0;

        for (input, target) in test_inputs.iter().zip(test_targets.iter()) {
            let output = self.forward(input)?;

            // Find max indices (for classification)
            let target_max_idx = target.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            let output_max_idx = output.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            if target_max_idx == output_max_idx {
                correct += 1;
            } else {
                incorrect += 1;
            }
        }

        let total = correct + incorrect;
        let accuracy = (correct as f32 / total as f32) * 100.0;

        Ok(TestResults { correct, incorrect, accuracy })
    }
}
```

**Reference:** `FFN.cpp:31-55`

#### 4.2: Implement File I/O

**File:** `src/utils/file_io.rs`

Port from `utility_f.cpp` (readMatFromFile):

```rust
use std::fs::File;
use std::io::{BufRead, BufReader};
use crate::Result;

pub fn read_matrix_from_file(path: &str) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut matrix = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let row: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<f32>().ok())
            .collect();

        if !row.is_empty() {
            matrix.push(row);
        }
    }

    Ok(matrix)
}
```

**Reference:** `utility_f.cpp` (original C++ implementation)

**Test:**
```rust
#[test]
fn test_read_matrix() {
    // Create temporary test file
    use std::io::Write;
    let mut file = File::create("/tmp/test_matrix.txt").unwrap();
    writeln!(file, "1.0 2.0 3.0").unwrap();
    writeln!(file, "4.0 5.0 6.0").unwrap();

    let matrix = read_matrix_from_file("/tmp/test_matrix.txt").unwrap();
    assert_eq!(matrix.len(), 2);
    assert_eq!(matrix[0], vec![1.0, 2.0, 3.0]);
    assert_eq!(matrix[1], vec![4.0, 5.0, 6.0]);
}
```

#### 4.3: Create XOR Example

**File:** `examples/xor.rs`

Port from `main.cpp:7-17`:

```rust
use neural_network_rs::FeedForwardNetwork;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Training XOR network...\n");

    let mut network = FeedForwardNetwork::new(2, 4, 1);

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

    network.train_by_error(&inputs, &targets, 0.0001)?;

    println!("\nTesting XOR network:");
    for (input, expected) in inputs.iter().zip(targets.iter()) {
        let output = network.forward(input)?;
        println!(
            "XOR({:.0}, {:.0}) = {:.4} (expected {:.0})",
            input[0], input[1], output[0], expected[0]
        );
    }

    Ok(())
}
```

**Reference:** `main.cpp:7-17`

**Run:** `cargo run --example xor`

#### 4.4: Create Digit Recognition Example

**File:** `examples/digit_recognition.rs`

Port from `main.cpp:21-32`:

```rust
use neural_network_rs::{FeedForwardNetwork, utils::read_matrix_from_file};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Training digit recognition network...\n");

    let mut network = FeedForwardNetwork::new(55, 20, 10);

    println!("Loading training data...");
    let train_x = read_matrix_from_file("samples/Xapp.txt")?;
    let train_y = read_matrix_from_file("samples/TA.txt")?;

    println!("Loading test data...");
    let test_x = read_matrix_from_file("samples/Xtest.txt")?;
    let test_y = read_matrix_from_file("samples/TT.txt")?;

    println!("Training network (1000 iterations)...");
    network.train_by_iteration(&train_x, &train_y, 1000)?;

    println!("\nEvaluating on test set...");
    let results = network.test(&test_x, &test_y)?;

    println!("Correct: {}", results.correct);
    println!("Incorrect: {}", results.incorrect);
    println!("Accuracy: {:.2}%", results.accuracy);

    Ok(())
}
```

**Reference:** `main.cpp:21-32`

**Run:** `cargo run --example digit_recognition`

**Success Criteria:**
- [ ] XOR converges to correct solution
- [ ] Digit recognition achieves ≥90% accuracy
- [ ] File I/O works correctly
- [ ] Examples run without errors

---

## Phase 5: Polish & Documentation (2-3 days)

### Goals
- Complete API documentation
- Write comprehensive README
- Add benchmarks
- Final testing

### Tasks

#### 5.1: Document All Public APIs

Add rustdoc comments to all public items:

```rust
/// A feed-forward neural network with backpropagation.
///
/// This network supports a 3-layer architecture (input, hidden, output)
/// and uses the backpropagation algorithm for training.
///
/// # Examples
///
/// ```
/// use neural_network_rs::FeedForwardNetwork;
///
/// let mut network = FeedForwardNetwork::new(2, 4, 1);
/// let output = network.forward(&[0.5, 0.5]).unwrap();
/// ```
pub struct FeedForwardNetwork {
    // ...
}
```

**Command:** `cargo doc --open` (verify documentation)

#### 5.2: Write Comprehensive README

**File:** `README.md`

Include:
- Project description
- Quick start guide
- Installation instructions
- Usage examples (XOR, digit recognition)
- API overview
- Performance notes
- Comparison to C++ version
- Contributing guidelines
- License

#### 5.3: Create Benchmarks

**File:** `benches/training_benchmark.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use neural_network_rs::FeedForwardNetwork;

fn xor_training_benchmark(c: &mut Criterion) {
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    c.bench_function("xor_training_1000_iterations", |b| {
        b.iter(|| {
            let mut network = FeedForwardNetwork::new(2, 4, 1);
            network.train_by_iteration(
                black_box(&inputs),
                black_box(&targets),
                1000
            ).unwrap();
        });
    });
}

criterion_group!(benches, xor_training_benchmark);
criterion_main!(benches);
```

**Run:** `cargo bench`

#### 5.4: Integration Tests

**File:** `tests/integration_tests.rs`

```rust
use neural_network_rs::FeedForwardNetwork;

#[test]
fn test_xor_learning() {
    let mut network = FeedForwardNetwork::new(2, 4, 1);

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    network.train_by_error(&inputs, &targets, 0.01).unwrap();

    // Verify network learned XOR
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let output = network.forward(input).unwrap();
        let error = (output[0] - target[0]).abs();
        assert!(error < 0.2, "XOR({}, {}) error too high: {}", input[0], input[1], error);
    }
}

#[test]
fn test_digit_recognition() {
    use neural_network_rs::utils::read_matrix_from_file;

    let mut network = FeedForwardNetwork::new(55, 20, 10);

    let train_x = read_matrix_from_file("samples/Xapp.txt").unwrap();
    let train_y = read_matrix_from_file("samples/TA.txt").unwrap();
    let test_x = read_matrix_from_file("samples/Xtest.txt").unwrap();
    let test_y = read_matrix_from_file("samples/TT.txt").unwrap();

    network.train_by_iteration(&train_x, &train_y, 1000).unwrap();

    let results = network.test(&test_x, &test_y).unwrap();
    assert!(results.accuracy >= 90.0, "Accuracy too low: {:.2}%", results.accuracy);
}
```

**Success Criteria:**
- [ ] All public APIs documented
- [ ] `cargo doc` generates clean documentation
- [ ] README is comprehensive
- [ ] Benchmarks run successfully
- [ ] Integration tests pass

---

## Phase 6: Release Preparation (1-2 days)

### Tasks

#### 6.1: Final Checklist

- [ ] All tests passing (`cargo test`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Formatted (`cargo fmt`)
- [ ] Documentation complete (`cargo doc`)
- [ ] Examples work (`cargo run --example xor`, `cargo run --example digit_recognition`)
- [ ] Benchmarks run (`cargo bench`)
- [ ] README accurate
- [ ] LICENSE file present
- [ ] CHANGELOG.md created

#### 6.2: Performance Comparison

Create a comparison table:

| Metric | C++ | Rust | Notes |
|--------|-----|------|-------|
| XOR training (1000 iter) | X ms | Y ms | |
| Digit recognition (1000 iter) | X ms | Y ms | |
| Memory usage | X MB | Y MB | |
| Accuracy (digit) | 93.3% | Z% | |

#### 6.3: Optional: Publish to crates.io

```bash
cargo login
cargo publish --dry-run
cargo publish
```

---

## Migration Notes

### Key Differences to Handle

1. **No Circular References**: Rust's ownership prevents Layer → Network pointers. Solution: Pass necessary context as parameters.

2. **Borrow Checker**: Cannot mutably borrow network while iterating layers. Solution: Use `split_at_mut()` or clone data when needed.

3. **Error Handling**: C++ uses exceptions; Rust uses `Result`. Solution: Propagate errors with `?` operator.

4. **Random Number Generation**: Different RNG APIs. Solution: Use `rand` crate with similar distribution.

5. **Matrix Library**: C++ uses `vector<vector<float>>`; Rust uses `ndarray`. Solution: Map operations to ndarray API.

### Testing Strategy

Each phase should include:
- Unit tests for new functionality
- Integration tests for complete features
- Comparison tests against C++ output (same inputs → same outputs)

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Performance regression | Profile and benchmark; use BLAS if needed |
| API design issues | Review with Rust community early |
| Borrow checker conflicts | Refactor to use clearer ownership |
| Numerical differences | Use epsilon comparison, document precision |

---

## Success Criteria

### Functional
- [ ] XOR learning works (converges in <10k iterations)
- [ ] Digit recognition ≥90% accuracy
- [ ] All examples run successfully
- [ ] File I/O works with sample data

### Quality
- [ ] Zero unsafe code
- [ ] 100% public API documented
- [ ] ≥80% test coverage
- [ ] No clippy warnings

### Performance
- [ ] Training speed within 10% of C++
- [ ] Memory usage reasonable
- [ ] No unnecessary allocations

---

## Timeline Summary

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 0: Setup | 2-3 days | Project structure, dependencies |
| 1: Data Structures | 3-4 days | Layer, Network, error types |
| 2: Forward Prop | 2-3 days | Forward pass working |
| 3: Backprop | 3-4 days | Training working |
| 4: Testing & Utils | 2-3 days | Examples, file I/O |
| 5: Polish | 2-3 days | Documentation, benchmarks |
| 6: Release | 1-2 days | Final checks, publish |
| **Total** | **15-22 days** | **Production-ready library** |

---

## Next Steps

1. Review this plan with team/stakeholders
2. Set up project repository
3. Begin Phase 0 (Project Setup)
4. Schedule weekly check-ins to track progress
5. Adjust timeline based on actual velocity

**Questions? Issues? Refer to:**
- `docs/architecture.md` for technical details
- `docs/PRD.md` for requirements
- Original C++ code for reference implementation
