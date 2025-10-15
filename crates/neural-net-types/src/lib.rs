//! # Neural Network Types
//!
//! Core data structures for neural network implementations. This crate provides
//! **pure data types** with no algorithms - just structures, serialization, and
//! error handling.
//!
//! ## Design Philosophy
//!
//! This crate contains ONLY data structures and type definitions. All algorithms
//! (forward propagation, backpropagation, training) live in `neural-net-core`.
//! All visualization logic lives in `neural-net-viz`.
//!
//! This separation ensures:
//! - No circular dependencies
//! - Clear separation of concerns
//! - Small, focused crate (< 500 LOC)
//! - Easy to understand and maintain
//!
//! ## Examples
//!
//! ```
//! use neural_net_types::{Layer, FeedForwardNetwork};
//!
//! // Create a network structure (no algorithms yet)
//! let network = FeedForwardNetwork::new(2, 4, 1);
//! assert_eq!(network.layer_count(), 3);
//!
//! // Access layers
//! let layer = network.layer(0).unwrap();
//! assert_eq!(layer.num_neurons(), 2);
//! ```

mod error;
mod layer;
mod metadata;
mod network;

pub use error::{NeuralNetError, Result};
pub use layer::Layer;
pub use metadata::{NetworkCheckpoint, NetworkMetadata};
pub use network::FeedForwardNetwork;
