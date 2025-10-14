//! # Neural Network Library
//!
//! A feed-forward neural network implementation with backpropagation training algorithm.
//!
//! This library provides a simple, type-safe implementation of a 3-layer neural network
//! (input, hidden, output) suitable for learning basic patterns and classification tasks.
//!
//! ## Examples
//!
//! ```
//! use neural_network_rs::FeedForwardNetwork;
//!
//! // Create a network: 2 inputs, 4 hidden neurons, 1 output
//! let network = FeedForwardNetwork::new(2, 4, 1);
//! assert_eq!(network.layer_count(), 3);
//! ```
//!
//! Full training example (when implementation is complete):
//!
//! ```ignore
//! use neural_network_rs::FeedForwardNetwork;
//!
//! let mut network = FeedForwardNetwork::new(2, 4, 1);
//!
//! // Training data for XOR
//! let inputs = vec![
//!     vec![0.0, 0.0],
//!     vec![0.0, 1.0],
//!     vec![1.0, 0.0],
//!     vec![1.0, 1.0],
//! ];
//! let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
//!
//! // Train the network
//! network.train_by_error(&inputs, &targets, 0.0001).unwrap();
//!
//! // Test the network
//! let output = network.forward(&[1.0, 0.0]).unwrap();
//! assert!((output[0] - 1.0).abs() < 0.1);
//! ```

mod activation;
mod error;
mod layer;
mod network;
pub mod utils;

pub use activation::{Activation, Linear, Sigmoid};
pub use error::{NeuralNetError, Result};
pub use network::{FeedForwardNetwork, TestResults};
