//! Neural network layer implementation

use crate::{NeuralNetError, Result};
use ndarray::Array2;
use rand::Rng;

/// A single layer in the neural network
///
/// Each layer (except the input layer) has weights connecting it to the previous layer,
/// and maintains its inputs, outputs, and deltas for backpropagation.
#[derive(Debug)]
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
    /// Create a new layer
    ///
    /// # Arguments
    ///
    /// * `index` - Layer index in the network
    /// * `num_neurons` - Number of neurons in this layer
    /// * `prev_layer_size` - Size of previous layer (None for input layer)
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

    /// Get the layer's outputs
    pub fn outputs(&self) -> &[f32] {
        &self.outputs
    }

    /// Get the layer's inputs
    pub fn inputs(&self) -> &[f32] {
        &self.inputs
    }

    /// Get the layer's deltas
    pub fn deltas(&self) -> &[f32] {
        &self.deltas
    }

    /// Get the layer's weights
    pub fn weights(&self) -> Option<&Array2<f32>> {
        self.weights.as_ref()
    }

    /// Set the layer's inputs directly (for input layer)
    pub fn set_inputs(&mut self, inputs: Vec<f32>) {
        self.inputs = inputs;
    }

    /// Forward propagation (to be implemented in Phase 2)
    pub fn forward_propagate(
        &mut self,
        prev_outputs: Option<&[f32]>,
        is_output_layer: bool,
    ) -> Result<()> {
        // TODO: Implement in Phase 2
        self.calc_inputs(prev_outputs)?;
        self.calc_outputs(is_output_layer);
        Ok(())
    }

    /// Calculate layer inputs from previous layer outputs (to be implemented)
    fn calc_inputs(&mut self, prev_outputs: Option<&[f32]>) -> Result<()> {
        if self.index == 0 {
            return Ok(()); // Input layer has no calculation
        }

        let prev_outputs = prev_outputs.ok_or_else(|| {
            NeuralNetError::InvalidConfig("Missing previous layer outputs".to_string())
        })?;

        let weights = self.weights.as_ref().ok_or_else(|| {
            NeuralNetError::InvalidConfig("No weights for non-input layer".to_string())
        })?;

        if weights.shape()[0] != prev_outputs.len() {
            return Err(NeuralNetError::DimensionMismatch {
                expected: weights.shape()[0],
                actual: prev_outputs.len(),
            });
        }

        // TODO: Implement matrix multiplication in Phase 2
        self.inputs.clear();
        for _ in 0..self.num_neurons {
            self.inputs.push(0.0);
        }

        Ok(())
    }

    /// Calculate layer outputs with activation function (to be implemented)
    fn calc_outputs(&mut self, _is_output_layer: bool) {
        // TODO: Implement in Phase 2
        self.outputs.clear();
        self.outputs.extend_from_slice(&self.inputs);
    }

    /// Calculate deltas for backpropagation (to be implemented in Phase 3)
    pub fn calc_deltas(
        &mut self,
        _targets: Option<&[f32]>,
        _next_layer: Option<&Layer>,
    ) -> Result<()> {
        // TODO: Implement in Phase 3
        Ok(())
    }

    /// Update weights using calculated deltas (to be implemented in Phase 3)
    pub fn update_weights(&mut self, _prev_outputs: &[f32], _learning_rate: f32) -> Result<()> {
        // TODO: Implement in Phase 3
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_creation_input() {
        let layer = Layer::new(0, 3, None);
        assert_eq!(layer.index, 0);
        assert_eq!(layer.num_neurons, 3);
        assert!(layer.weights.is_none());
    }

    #[test]
    fn test_layer_creation_hidden() {
        let layer = Layer::new(1, 4, Some(3));
        assert_eq!(layer.index, 1);
        assert_eq!(layer.num_neurons, 4);
        assert!(layer.weights.is_some());

        let weights = layer.weights.unwrap();
        assert_eq!(weights.shape(), &[3, 4]);
    }

    #[test]
    fn test_weight_initialization_range() {
        let layer = Layer::new(1, 10, Some(5));
        let weights = layer.weights.unwrap();

        for &w in weights.iter() {
            assert!(w >= -1.0 && w <= 1.0, "Weight {} out of range", w);
        }
    }
}
