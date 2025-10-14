//! Feed-forward neural network implementation

use crate::layer::Layer;
use crate::{NeuralNetError, Result};
use serde::{Deserialize, Serialize};

/// A feed-forward neural network with backpropagation
///
/// This network supports a 3-layer architecture (input, hidden, output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForwardNetwork {
    /// Network layers
    layers: Vec<Layer>,
    /// Current training targets (will be used in Phase 3)
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
    /// use neural_net_core::FeedForwardNetwork;
    ///
    /// let network = FeedForwardNetwork::new(2, 4, 1);
    /// assert_eq!(network.layer_count(), 3);
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

    /// Get the number of layers
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Get a reference to a layer
    pub fn layer(&self, index: usize) -> Option<&Layer> {
        self.layers.get(index)
    }

    /// Get a mutable reference to a layer (for manual weight tuning examples)
    pub fn layer_mut(&mut self, index: usize) -> Option<&mut Layer> {
        self.layers.get_mut(index)
    }

    /// Forward propagation through the network
    pub fn forward(&mut self, inputs: &[f32]) -> Result<Vec<f32>> {
        // Validate input size
        if inputs.len() != self.layers[0].num_neurons() {
            return Err(NeuralNetError::DimensionMismatch {
                expected: self.layers[0].num_neurons(),
                actual: inputs.len(),
            });
        }

        // Set inputs on input layer
        self.layers[0].set_inputs(inputs.to_vec());
        self.layers[0].forward_propagate(None, false)?;

        // Propagate through hidden layer(s)
        for i in 1..self.layers.len() - 1 {
            let prev_outputs = self.layers[i - 1].outputs().to_vec();
            self.layers[i].forward_propagate(Some(&prev_outputs), false)?;
        }

        // Propagate through output layer
        let output_idx = self.layers.len() - 1;
        let prev_outputs = self.layers[output_idx - 1].outputs().to_vec();
        self.layers[output_idx].forward_propagate(Some(&prev_outputs), true)?;

        // Return output layer's outputs
        Ok(self.layers[output_idx].outputs().to_vec())
    }

    /// Train by iteration count (to be implemented in Phase 3)
    pub fn train_by_iteration(
        &mut self,
        _inputs: &[Vec<f32>],
        _targets: &[Vec<f32>],
        _iterations: usize,
    ) -> Result<()> {
        // TODO: Implement in Phase 3
        Ok(())
    }

    /// Train until error threshold is reached (to be implemented in Phase 3)
    pub fn train_by_error(
        &mut self,
        _inputs: &[Vec<f32>],
        _targets: &[Vec<f32>],
        _target_error: f32,
    ) -> Result<()> {
        // TODO: Implement in Phase 3
        Ok(())
    }

    /// Test the network on a dataset (to be implemented in Phase 4)
    pub fn test(
        &mut self,
        _test_inputs: &[Vec<f32>],
        _test_targets: &[Vec<f32>],
    ) -> Result<TestResults> {
        // TODO: Implement in Phase 4
        Ok(TestResults {
            correct: 0,
            incorrect: 0,
            accuracy: 0.0,
        })
    }
}

/// Results from testing a network
#[derive(Debug, Clone)]
pub struct TestResults {
    /// Number of correct predictions
    pub correct: usize,
    /// Number of incorrect predictions
    pub incorrect: usize,
    /// Accuracy percentage (0-100)
    pub accuracy: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let net = FeedForwardNetwork::new(2, 4, 1);
        assert_eq!(net.layer_count(), 3);

        let layer0 = net.layer(0).unwrap();
        assert_eq!(layer0.num_neurons(), 2);
        assert!(layer0.weights().is_none());

        let layer1 = net.layer(1).unwrap();
        assert_eq!(layer1.num_neurons(), 4);
        assert_eq!(layer1.weights().unwrap().shape(), &[2, 4]);

        let layer2 = net.layer(2).unwrap();
        assert_eq!(layer2.num_neurons(), 1);
        assert_eq!(layer2.weights().unwrap().shape(), &[4, 1]);
    }

    #[test]
    fn test_forward_dimension_validation() {
        let mut net = FeedForwardNetwork::new(2, 3, 1);
        let result = net.forward(&[0.5]); // Wrong size

        assert!(matches!(
            result,
            Err(NeuralNetError::DimensionMismatch { .. })
        ));
    }
}
