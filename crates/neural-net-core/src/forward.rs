//! Forward propagation algorithms
//!
//! Extension traits that add forward propagation capabilities to network types.

use crate::activation::{Activation, Linear, Sigmoid};
use neural_net_types::{FeedForwardNetwork, Layer, NeuralNetError, Result};

/// Extension trait for Layer forward propagation
pub trait LayerForward {
    /// Perform forward propagation through this layer
    ///
    /// # Arguments
    ///
    /// * `prev_outputs` - Outputs from previous layer (None for input layer)
    /// * `is_output_layer` - Whether this is the output layer (affects activation function)
    fn forward_propagate(&mut self, prev_outputs: Option<&[f32]>, is_output_layer: bool) -> Result<()>;
}

impl LayerForward for Layer {
    fn forward_propagate(&mut self, prev_outputs: Option<&[f32]>, is_output_layer: bool) -> Result<()> {
        calc_inputs(self, prev_outputs)?;
        calc_outputs(self, is_output_layer);
        Ok(())
    }
}

/// Calculate layer inputs from previous layer outputs
///
/// Performs matrix multiplication: inputs = weights^T * prev_outputs
/// For each neuron j in this layer:
///   inputs[j] = sum(prev_outputs[i] * weights[i][j]) for all i
fn calc_inputs(layer: &mut Layer, prev_outputs: Option<&[f32]>) -> Result<()> {
    if layer.index() == 0 {
        return Ok(()); // Input layer has no calculation
    }

    let prev_outputs = prev_outputs.ok_or_else(|| {
        NeuralNetError::InvalidConfig("Missing previous layer outputs".to_string())
    })?;

    // Compute all inputs first (immutable borrows only), then write (mutable borrow)
    // This avoids holding multiple borrows simultaneously
    let new_inputs: Vec<f32> = {
        let weights = layer.weights().ok_or_else(|| {
            NeuralNetError::InvalidConfig("No weights for non-input layer".to_string())
        })?;

        if weights.shape()[0] != prev_outputs.len() {
            return Err(NeuralNetError::DimensionMismatch {
                expected: weights.shape()[0],
                actual: prev_outputs.len(),
            });
        }

        let num_neurons = weights.shape()[1];
        let num_prev = weights.shape()[0];

        // Matrix multiplication: compute all neuron inputs
        (0..num_neurons)
            .map(|col| {
                (0..num_prev)
                    .map(|row| prev_outputs[row] * weights[[row, col]])
                    .sum()
            })
            .collect()
    }; // weights borrow ends here

    // Now write results with single mutable borrow
    *layer.inputs_mut() = new_inputs;

    Ok(())
}

/// Calculate layer outputs with activation function
///
/// Applies activation function to inputs:
/// - Input layer (index 0) and output layer: Linear activation (identity)
/// - Hidden layers: Sigmoid activation (1 / (1 + e^-x))
fn calc_outputs(layer: &mut Layer, is_output_layer: bool) {
    // Compute all outputs first (immutable borrows only), then write (mutable borrow)
    let new_outputs: Vec<f32> = {
        let layer_index = layer.index();
        let inputs = layer.inputs();

        if layer_index == 0 || is_output_layer {
            // Linear activation for input and output layers
            let linear = Linear;
            inputs.iter().map(|&input| linear.activate(input)).collect()
        } else {
            // Sigmoid activation for hidden layers
            let sigmoid = Sigmoid;
            inputs.iter().map(|&input| sigmoid.activate(input)).collect()
        }
    }; // immutable borrows end here

    // Now write results with single mutable borrow
    *layer.outputs_mut() = new_outputs;
}

/// Extension trait for FeedForwardNetwork forward propagation
pub trait ForwardPropagation {
    /// Perform forward propagation through the entire network
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input vector (must match input layer size)
    ///
    /// # Returns
    ///
    /// Output vector from the output layer
    ///
    /// # Errors
    ///
    /// Returns error if input dimensions don't match network input size
    ///
    /// # Examples
    ///
    /// ```
    /// use neural_net_core::{FeedForwardNetwork, ForwardPropagation};
    ///
    /// let mut network = FeedForwardNetwork::new(2, 4, 1);
    /// let output = network.forward(&[1.0, 0.0]).unwrap();
    /// assert_eq!(output.len(), 1);
    /// ```
    fn forward(&mut self, inputs: &[f32]) -> Result<Vec<f32>>;
}

impl ForwardPropagation for FeedForwardNetwork {
    fn forward(&mut self, inputs: &[f32]) -> Result<Vec<f32>> {
        // Validate input size
        if inputs.len() != self.layer(0).unwrap().num_neurons() {
            return Err(NeuralNetError::DimensionMismatch {
                expected: self.layer(0).unwrap().num_neurons(),
                actual: inputs.len(),
            });
        }

        // Set inputs on input layer
        self.layer_mut(0).unwrap().set_inputs(inputs.to_vec());
        self.layer_mut(0).unwrap().forward_propagate(None, false)?;

        // Propagate through hidden layer(s)
        for i in 1..self.layer_count() - 1 {
            let prev_outputs = self.layer(i - 1).unwrap().outputs().to_vec();
            self.layer_mut(i).unwrap().forward_propagate(Some(&prev_outputs), false)?;
        }

        // Propagate through output layer
        let output_idx = self.layer_count() - 1;
        let prev_outputs = self.layer(output_idx - 1).unwrap().outputs().to_vec();
        self.layer_mut(output_idx).unwrap().forward_propagate(Some(&prev_outputs), true)?;

        // Return output layer's outputs
        Ok(self.layer(output_idx).unwrap().outputs().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_forward_propagate_input_layer() {
        let mut layer = Layer::new(0, 3, None);
        layer.set_inputs(vec![0.5, -0.3, 0.8]);

        layer.forward_propagate(None, false).unwrap();

        // Input layer should have outputs == inputs (linear activation)
        assert_eq!(layer.outputs(), &[0.5, -0.3, 0.8]);
    }

    #[test]
    fn test_forward_propagate_hidden_layer() {
        let mut layer = Layer::new(1, 2, Some(3));

        // Set known weights for deterministic testing
        layer.set_weights(Array2::from_shape_vec(
            (3, 2),
            vec![
                0.5, 0.2,  // weights from prev neuron 0 to current neurons [0, 1]
                0.3, 0.4,  // weights from prev neuron 1 to current neurons [0, 1]
                0.1, 0.6,  // weights from prev neuron 2 to current neurons [0, 1]
            ],
        ).unwrap()).unwrap();

        let prev_outputs = vec![1.0, 0.5, 0.2];

        layer.forward_propagate(Some(&prev_outputs), false).unwrap();

        // Expected inputs:
        // neuron 0: 1.0*0.5 + 0.5*0.3 + 0.2*0.1 = 0.5 + 0.15 + 0.02 = 0.67
        // neuron 1: 1.0*0.2 + 0.5*0.4 + 0.2*0.6 = 0.2 + 0.2 + 0.12 = 0.52
        assert_eq!(layer.inputs().len(), 2);
        assert!((layer.inputs()[0] - 0.67).abs() < 1e-6);
        assert!((layer.inputs()[1] - 0.52).abs() < 1e-6);

        // Outputs should be sigmoid(inputs) for hidden layer
        // sigmoid(0.67) ≈ 0.6616
        // sigmoid(0.52) ≈ 0.6271
        assert_eq!(layer.outputs().len(), 2);
        assert!((layer.outputs()[0] - 0.6616).abs() < 1e-3);
        assert!((layer.outputs()[1] - 0.6271).abs() < 1e-3);
    }

    #[test]
    fn test_forward_propagate_output_layer() {
        let mut layer = Layer::new(2, 2, Some(3));

        layer.set_weights(Array2::from_shape_vec(
            (3, 2),
            vec![0.5, 0.3, 0.2, 0.4, 0.1, 0.6],
        ).unwrap()).unwrap();

        let prev_outputs = vec![0.8, 0.6, 0.4];

        layer.forward_propagate(Some(&prev_outputs), true).unwrap();

        // Expected inputs:
        // neuron 0: 0.8*0.5 + 0.6*0.2 + 0.4*0.1 = 0.4 + 0.12 + 0.04 = 0.56
        // neuron 1: 0.8*0.3 + 0.6*0.4 + 0.4*0.6 = 0.24 + 0.24 + 0.24 = 0.72
        assert_eq!(layer.inputs().len(), 2);
        assert!((layer.inputs()[0] - 0.56).abs() < 1e-6);
        assert!((layer.inputs()[1] - 0.72).abs() < 1e-6);

        // Outputs should be linear (inputs == outputs) for output layer
        assert_eq!(layer.outputs(), layer.inputs());
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

    #[test]
    fn test_network_forward() {
        let mut network = FeedForwardNetwork::new(2, 3, 1);
        let output = network.forward(&[1.0, 0.0]).unwrap();
        assert_eq!(output.len(), 1);
    }

    #[test]
    fn test_network_forward_dimension_validation() {
        let mut network = FeedForwardNetwork::new(2, 3, 1);
        let result = network.forward(&[0.5]); // Wrong size

        assert!(matches!(
            result,
            Err(NeuralNetError::DimensionMismatch { .. })
        ));
    }
}
