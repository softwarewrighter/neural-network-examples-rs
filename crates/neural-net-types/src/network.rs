//! Feed-forward neural network data structure
//!
//! This module contains ONLY the FeedForwardNetwork struct and basic accessors.
//! All algorithm logic (forward propagation, backpropagation, training) lives in neural-net-core.

use crate::Layer;
use serde::{Deserialize, Serialize};

/// A feed-forward neural network data structure
///
/// This network supports a 3-layer architecture (input, hidden, output).
///
/// **Note:** This is a pure data structure. For algorithms (forward/backward propagation,
/// training), see the `neural-net-core` crate.
///
/// # Examples
///
/// ```
/// use neural_net_types::FeedForwardNetwork;
///
/// let network = FeedForwardNetwork::new(2, 4, 1);
/// assert_eq!(network.layer_count(), 3);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForwardNetwork {
    /// Network layers (input, hidden, output)
    layers: Vec<Layer>,
    /// Current training targets (used during training)
    #[allow(dead_code)]
    targets: Option<Vec<f32>>,
}

impl FeedForwardNetwork {
    /// Create a new 3-layer feed-forward network
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input neurons
    /// * `hidden_size` - Number of hidden layer neurons
    /// * `output_size` - Number of output neurons
    ///
    /// # Examples
    ///
    /// ```
    /// use neural_net_types::FeedForwardNetwork;
    ///
    /// let network = FeedForwardNetwork::new(2, 4, 1);
    /// assert_eq!(network.layer_count(), 3);
    ///
    /// // Verify layer sizes
    /// assert_eq!(network.layer(0).unwrap().num_neurons(), 2);
    /// assert_eq!(network.layer(1).unwrap().num_neurons(), 4);
    /// assert_eq!(network.layer(2).unwrap().num_neurons(), 1);
    /// ```
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let layers = vec![
            Layer::new(0, input_size, None),               // Input layer
            Layer::new(1, hidden_size, Some(input_size)),  // Hidden layer
            Layer::new(2, output_size, Some(hidden_size)), // Output layer
        ];

        Self {
            layers,
            targets: None,
        }
    }

    /// Get the number of layers in the network
    ///
    /// # Examples
    ///
    /// ```
    /// use neural_net_types::FeedForwardNetwork;
    ///
    /// let network = FeedForwardNetwork::new(2, 4, 1);
    /// assert_eq!(network.layer_count(), 3);
    /// ```
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Get an immutable reference to a layer by index
    ///
    /// # Arguments
    ///
    /// * `index` - Layer index (0 = input, 1 = hidden, 2 = output)
    ///
    /// # Returns
    ///
    /// `Some(&Layer)` if index is valid, `None` otherwise
    pub fn layer(&self, index: usize) -> Option<&Layer> {
        self.layers.get(index)
    }

    /// Get a mutable reference to a layer by index
    ///
    /// Useful for manual weight tuning in examples and testing.
    ///
    /// # Arguments
    ///
    /// * `index` - Layer index (0 = input, 1 = hidden, 2 = output)
    ///
    /// # Returns
    ///
    /// `Some(&mut Layer)` if index is valid, `None` otherwise
    pub fn layer_mut(&mut self, index: usize) -> Option<&mut Layer> {
        self.layers.get_mut(index)
    }

    /// Get immutable reference to all layers
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// Get mutable reference to all layers
    pub fn layers_mut(&mut self) -> &mut [Layer] {
        &mut self.layers
    }

    /// Get the current training targets
    pub fn targets(&self) -> Option<&Vec<f32>> {
        self.targets.as_ref()
    }

    /// Set the current training targets
    pub fn set_targets(&mut self, targets: Option<Vec<f32>>) {
        self.targets = targets;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let network = FeedForwardNetwork::new(2, 4, 1);
        assert_eq!(network.layer_count(), 3);

        let layer0 = network.layer(0).unwrap();
        assert_eq!(layer0.num_neurons(), 2);
        assert!(layer0.weights().is_none());

        let layer1 = network.layer(1).unwrap();
        assert_eq!(layer1.num_neurons(), 4);
        assert_eq!(layer1.weights().unwrap().shape(), &[2, 4]);

        let layer2 = network.layer(2).unwrap();
        assert_eq!(layer2.num_neurons(), 1);
        assert_eq!(layer2.weights().unwrap().shape(), &[4, 1]);
    }

    #[test]
    fn test_layer_access() {
        let network = FeedForwardNetwork::new(3, 5, 2);

        // Valid indices
        assert!(network.layer(0).is_some());
        assert!(network.layer(1).is_some());
        assert!(network.layer(2).is_some());

        // Invalid index
        assert!(network.layer(3).is_none());
    }

    #[test]
    fn test_layer_mut() {
        let mut network = FeedForwardNetwork::new(2, 3, 1);

        // Get mutable reference and modify
        if let Some(layer) = network.layer_mut(0) {
            layer.set_inputs(vec![1.0, 2.0]);
        }

        // Verify modification
        assert_eq!(network.layer(0).unwrap().inputs(), &[1.0, 2.0]);
    }

    #[test]
    fn test_targets() {
        let mut network = FeedForwardNetwork::new(2, 3, 1);

        // Initially None
        assert!(network.targets().is_none());

        // Set targets
        network.set_targets(Some(vec![0.5, 0.7]));
        assert_eq!(network.targets(), Some(&vec![0.5, 0.7]));

        // Clear targets
        network.set_targets(None);
        assert!(network.targets().is_none());
    }
}
