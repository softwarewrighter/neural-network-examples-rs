//! # Neural Network Algorithms
//!
//! This crate provides algorithms for neural network operations (forward propagation,
//! backpropagation, training). It depends on `neural-net-types` for data structures.
//!
//! ## Architecture
//!
//! - **Data structures** (Layer, Network, errors) → `neural-net-types` crate
//! - **Algorithms** (forward/backward propagation) → this crate (`neural-net-core`)
//! - **Visualization** (SVG generation) → `neural-net-viz` crate
//!
//! This separation prevents circular dependencies and keeps each crate small and focused.
//!
//! ## Examples
//!
//! ```
//! use neural_net_core::{FeedForwardNetwork, ForwardPropagation};
//!
//! // Create a network: 2 inputs, 4 hidden neurons, 1 output
//! let mut network = FeedForwardNetwork::new(2, 4, 1);
//! assert_eq!(network.layer_count(), 3);
//!
//! // Forward propagation
//! let output = network.forward(&[1.0, 0.0]).unwrap();
//! assert_eq!(output.len(), 1);
//! ```

mod activation;
mod backward;
mod forward;
pub mod optimizer;
pub mod utils;

// Re-export all types from neural-net-types for convenience
pub use neural_net_types::{
    FeedForwardNetwork, Layer, NetworkCheckpoint, NetworkMetadata, NeuralNetError, Result,
};

// Export activation functions
pub use activation::{Activation, GELU, LeakyReLU, Linear, ReLU, Sigmoid, Swish, Tanh};

// Export algorithm traits
pub use backward::{LayerBackward, NetworkTraining};
pub use forward::ForwardPropagation;

// Export optimizers
pub use optimizer::{Adam, AdamW, Optimizer, RMSprop, SGD, SGDMomentum};
