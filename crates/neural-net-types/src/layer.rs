//! Neural network layer data structure
//!
//! This module contains ONLY the Layer struct and basic accessors/mutators.
//! All algorithm logic (forward propagation, backpropagation) lives in neural-net-core.

use crate::{NeuralNetError, Result};
use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// A single layer in the neural network
///
/// Each layer (except the input layer) has weights connecting it to the previous layer,
/// and maintains its inputs, outputs, and deltas for backpropagation.
///
/// **Note:** This is a pure data structure. For algorithms (forward/backward propagation),
/// see the `neural-net-core` crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// Layer index (0 = input, 1 = hidden, 2 = output for 3-layer network)
    pub(crate) index: usize,
    /// Number of neurons in this layer
    pub(crate) num_neurons: usize,
    /// Weight matrix: [prev_layer_size x num_neurons]
    /// None for input layer
    pub(crate) weights: Option<Array2<f32>>,
    /// Input values to this layer
    pub(crate) inputs: Vec<f32>,
    /// Output values from this layer (after activation)
    pub(crate) outputs: Vec<f32>,
    /// Delta values for backpropagation
    pub(crate) deltas: Vec<f32>,
}

impl Layer {
    /// Create a new layer with random weight initialization
    ///
    /// # Arguments
    ///
    /// * `index` - Layer index in the network
    /// * `num_neurons` - Number of neurons in this layer
    /// * `prev_layer_size` - Size of previous layer (None for input layer)
    ///
    /// # Examples
    ///
    /// ```
    /// use neural_net_types::Layer;
    ///
    /// // Input layer (no weights)
    /// let input_layer = Layer::new(0, 2, None);
    /// assert_eq!(input_layer.num_neurons(), 2);
    /// assert!(input_layer.weights().is_none());
    ///
    /// // Hidden layer with weights
    /// let hidden_layer = Layer::new(1, 4, Some(2));
    /// assert_eq!(hidden_layer.num_neurons(), 4);
    /// assert!(hidden_layer.weights().is_some());
    /// ```
    pub fn new(index: usize, num_neurons: usize, prev_layer_size: Option<usize>) -> Self {
        let weights = prev_layer_size.map(|prev_size| {
            let mut rng = rand::thread_rng();
            Array2::from_shape_fn((prev_size, num_neurons), |_| rng.gen_range(-1.0..1.0))
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

    /// Get the layer index
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the number of neurons in this layer
    pub fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    /// Get the layer's weights (immutable)
    pub fn weights(&self) -> Option<&Array2<f32>> {
        self.weights.as_ref()
    }

    /// Get the layer's inputs (immutable)
    pub fn inputs(&self) -> &[f32] {
        &self.inputs
    }

    /// Get the layer's outputs (immutable)
    pub fn outputs(&self) -> &[f32] {
        &self.outputs
    }

    /// Get the layer's deltas (immutable)
    pub fn deltas(&self) -> &[f32] {
        &self.deltas
    }

    /// Set the layer's weights (useful for testing and examples)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Attempting to set weights on input layer (index 0)
    /// - Weight dimensions don't match expected shape
    pub fn set_weights(&mut self, weights: Array2<f32>) -> Result<()> {
        if self.index == 0 {
            return Err(NeuralNetError::InvalidConfig(
                "Cannot set weights on input layer".to_string(),
            ));
        }

        let expected_shape = (
            self.weights
                .as_ref()
                .ok_or_else(|| NeuralNetError::InvalidConfig("No weights initialized".to_string()))?
                .shape()[0],
            self.num_neurons,
        );

        if weights.shape() != [expected_shape.0, expected_shape.1] {
            return Err(NeuralNetError::DimensionMismatch {
                expected: expected_shape.0 * expected_shape.1,
                actual: weights.len(),
            });
        }

        self.weights = Some(weights);
        Ok(())
    }

    /// Set the layer's inputs directly (used by algorithms in neural-net-core)
    pub fn set_inputs(&mut self, inputs: Vec<f32>) {
        self.inputs = inputs;
    }

    /// Set the layer's outputs directly (used by algorithms in neural-net-core)
    pub fn set_outputs(&mut self, outputs: Vec<f32>) {
        self.outputs = outputs;
    }

    /// Set the layer's deltas directly (used by algorithms in neural-net-core)
    pub fn set_deltas(&mut self, deltas: Vec<f32>) {
        self.deltas = deltas;
    }

    /// Get mutable reference to inputs (used by algorithms in neural-net-core)
    pub fn inputs_mut(&mut self) -> &mut Vec<f32> {
        &mut self.inputs
    }

    /// Get mutable reference to outputs (used by algorithms in neural-net-core)
    pub fn outputs_mut(&mut self) -> &mut Vec<f32> {
        &mut self.outputs
    }

    /// Get mutable reference to deltas (used by algorithms in neural-net-core)
    pub fn deltas_mut(&mut self) -> &mut Vec<f32> {
        &mut self.deltas
    }

    /// Get mutable reference to weights (used by algorithms in neural-net-core)
    pub fn weights_mut(&mut self) -> Option<&mut Array2<f32>> {
        self.weights.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation_input() {
        let layer = Layer::new(0, 3, None);
        assert_eq!(layer.index(), 0);
        assert_eq!(layer.num_neurons(), 3);
        assert!(layer.weights().is_none());
    }

    #[test]
    fn test_layer_creation_hidden() {
        let layer = Layer::new(1, 4, Some(3));
        assert_eq!(layer.index(), 1);
        assert_eq!(layer.num_neurons(), 4);
        assert!(layer.weights().is_some());

        let weights = layer.weights().unwrap();
        assert_eq!(weights.shape(), &[3, 4]);
    }

    #[test]
    fn test_weight_initialization_range() {
        let layer = Layer::new(1, 10, Some(5));
        let weights = layer.weights().unwrap();

        for &w in weights.iter() {
            assert!(w >= -1.0 && w <= 1.0, "Weight {} out of range", w);
        }
    }

    #[test]
    fn test_set_weights_validation() {
        let mut layer = Layer::new(1, 2, Some(3));

        // Valid weights
        let valid_weights = Array2::from_shape_vec((3, 2), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
        assert!(layer.set_weights(valid_weights).is_ok());

        // Invalid dimension
        let invalid_weights = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        assert!(layer.set_weights(invalid_weights).is_err());
    }

    #[test]
    fn test_cannot_set_weights_on_input_layer() {
        let mut layer = Layer::new(0, 3, None);
        let weights = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();

        let result = layer.set_weights(weights);
        assert!(result.is_err());
        assert!(matches!(result, Err(NeuralNetError::InvalidConfig(_))));
    }

    #[test]
    fn test_accessors_and_mutators() {
        let mut layer = Layer::new(1, 2, Some(3));

        // Test set_inputs
        layer.set_inputs(vec![1.0, 2.0]);
        assert_eq!(layer.inputs(), &[1.0, 2.0]);

        // Test set_outputs
        layer.set_outputs(vec![0.5, 0.7]);
        assert_eq!(layer.outputs(), &[0.5, 0.7]);

        // Test set_deltas
        layer.set_deltas(vec![0.1, 0.2]);
        assert_eq!(layer.deltas(), &[0.1, 0.2]);
    }
}
