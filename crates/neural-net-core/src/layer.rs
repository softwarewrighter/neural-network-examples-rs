//! Neural network layer implementation

use crate::{NeuralNetError, Result};
use ndarray::Array2;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// A single layer in the neural network
///
/// Each layer (except the input layer) has weights connecting it to the previous layer,
/// and maintains its inputs, outputs, and deltas for backpropagation.
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

    /// Get the number of neurons in this layer
    pub fn num_neurons(&self) -> usize {
        self.num_neurons
    }

    /// Get the layer's weights
    pub fn weights(&self) -> Option<&Array2<f32>> {
        self.weights.as_ref()
    }

    /// Set the layer's weights (useful for testing and examples)
    pub fn set_weights(&mut self, weights: Array2<f32>) -> Result<()> {
        if self.index == 0 {
            return Err(NeuralNetError::InvalidConfig(
                "Cannot set weights on input layer".to_string(),
            ));
        }

        let expected_shape = (
            self.weights.as_ref()
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

    /// Calculate layer inputs from previous layer outputs
    ///
    /// Performs matrix multiplication: inputs = weights^T * prev_outputs
    /// For each neuron j in this layer:
    ///   inputs[j] = sum(prev_outputs[i] * weights[i][j]) for all i
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

        // Matrix multiplication: for each neuron (column) in this layer
        self.inputs.clear();
        for col in 0..self.num_neurons {
            let mut sum = 0.0;
            for row in 0..prev_outputs.len() {
                sum += prev_outputs[row] * weights[[row, col]];
            }
            self.inputs.push(sum);
        }

        Ok(())
    }

    /// Calculate layer outputs with activation function
    ///
    /// Applies activation function to inputs:
    /// - Input layer (index 0) and output layer: Linear activation (identity)
    /// - Hidden layers: Sigmoid activation (1 / (1 + e^-x))
    fn calc_outputs(&mut self, is_output_layer: bool) {
        use crate::activation::{Activation, Linear, Sigmoid};

        self.outputs.clear();

        if self.index == 0 || is_output_layer {
            // Linear activation for input and output layers
            let linear = Linear;
            for &input in &self.inputs {
                self.outputs.push(linear.activate(input));
            }
        } else {
            // Sigmoid activation for hidden layers
            let sigmoid = Sigmoid;
            for &input in &self.inputs {
                self.outputs.push(sigmoid.activate(input));
            }
        }
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

    #[test]
    fn test_forward_propagate_input_layer() {
        let mut layer = Layer::new(0, 3, None);
        layer.inputs = vec![0.5, -0.3, 0.8];

        layer.forward_propagate(None, false).unwrap();

        // Input layer should have outputs == inputs (linear activation)
        assert_eq!(layer.outputs, vec![0.5, -0.3, 0.8]);
    }

    #[test]
    fn test_forward_propagate_hidden_layer() {
        use ndarray::Array2;

        let mut layer = Layer::new(1, 2, Some(3));

        // Set known weights for deterministic testing
        // weights[prev_neuron][current_neuron]
        layer.weights = Some(Array2::from_shape_vec(
            (3, 2),
            vec![
                0.5, 0.2,  // weights from prev neuron 0 to current neurons [0, 1]
                0.3, 0.4,  // weights from prev neuron 1 to current neurons [0, 1]
                0.1, 0.6,  // weights from prev neuron 2 to current neurons [0, 1]
            ],
        ).unwrap());

        let prev_outputs = vec![1.0, 0.5, 0.2];

        layer.forward_propagate(Some(&prev_outputs), false).unwrap();

        // Expected inputs:
        // neuron 0: 1.0*0.5 + 0.5*0.3 + 0.2*0.1 = 0.5 + 0.15 + 0.02 = 0.67
        // neuron 1: 1.0*0.2 + 0.5*0.4 + 0.2*0.6 = 0.2 + 0.2 + 0.12 = 0.52
        assert_eq!(layer.inputs.len(), 2);
        assert!((layer.inputs[0] - 0.67).abs() < 1e-6);
        assert!((layer.inputs[1] - 0.52).abs() < 1e-6);

        // Outputs should be sigmoid(inputs) for hidden layer
        // sigmoid(0.67) ≈ 0.6616
        // sigmoid(0.52) ≈ 0.6271
        assert_eq!(layer.outputs.len(), 2);
        assert!((layer.outputs[0] - 0.6616).abs() < 1e-3);
        assert!((layer.outputs[1] - 0.6271).abs() < 1e-3);
    }

    #[test]
    fn test_forward_propagate_output_layer() {
        use ndarray::Array2;

        let mut layer = Layer::new(2, 2, Some(3));

        layer.weights = Some(Array2::from_shape_vec(
            (3, 2),
            vec![0.5, 0.3, 0.2, 0.4, 0.1, 0.6],
        ).unwrap());

        let prev_outputs = vec![0.8, 0.6, 0.4];

        layer.forward_propagate(Some(&prev_outputs), true).unwrap();

        // Expected inputs:
        // neuron 0: 0.8*0.5 + 0.6*0.2 + 0.4*0.1 = 0.4 + 0.12 + 0.04 = 0.56
        // neuron 1: 0.8*0.3 + 0.6*0.4 + 0.4*0.6 = 0.24 + 0.24 + 0.24 = 0.72
        assert_eq!(layer.inputs.len(), 2);
        assert!((layer.inputs[0] - 0.56).abs() < 1e-6);
        assert!((layer.inputs[1] - 0.72).abs() < 1e-6);

        // Outputs should be linear (inputs == outputs) for output layer
        assert_eq!(layer.outputs, layer.inputs);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut layer = Layer::new(1, 2, Some(3));
        let prev_outputs = vec![1.0, 0.5]; // Wrong size (expected 3)

        let result = layer.forward_propagate(Some(&prev_outputs), false);

        assert!(result.is_err());
        match result {
            Err(NeuralNetError::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 2);
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }
}
