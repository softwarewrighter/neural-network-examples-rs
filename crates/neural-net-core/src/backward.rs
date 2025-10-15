//! Backpropagation algorithms
//!
//! Extension traits that add backpropagation capabilities to network types.

use neural_net_types::{Layer, NeuralNetError, Result};

/// Extension trait for Layer backpropagation
pub trait LayerBackward {
    /// Calculate deltas for this layer during backpropagation
    ///
    /// # Arguments
    ///
    /// * `targets` - Target values (only for output layer)
    /// * `next_layer_deltas` - Deltas from next layer (only for hidden layers)
    /// * `next_layer_weights` - Weights from next layer (only for hidden layers)
    fn calc_deltas(
        &mut self,
        targets: Option<&[f32]>,
        next_layer_deltas: Option<&[f32]>,
        next_layer_weights: Option<&ndarray::Array2<f32>>,
    ) -> Result<()>;

    /// Update weights based on deltas and learning rate
    ///
    /// # Arguments
    ///
    /// * `prev_outputs` - Outputs from previous layer
    /// * `learning_rate` - Learning rate (typically 0.01)
    fn update_weights(&mut self, prev_outputs: &[f32], learning_rate: f32) -> Result<()>;
}

impl LayerBackward for Layer {
    fn calc_deltas(
        &mut self,
        targets: Option<&[f32]>,
        next_layer_deltas: Option<&[f32]>,
        next_layer_weights: Option<&ndarray::Array2<f32>>,
    ) -> Result<()> {
        if self.index() == 0 {
            return Ok(()); // Input layer has no deltas
        }

        calc_deltas_impl(
            self,
            targets,
            next_layer_deltas,
            next_layer_weights,
        )
    }

    fn update_weights(&mut self, prev_outputs: &[f32], learning_rate: f32) -> Result<()> {
        if self.index() == 0 {
            return Ok(()); // Input layer has no weights
        }

        update_weights_impl(self, prev_outputs, learning_rate)
    }
}

/// Calculate deltas for a layer
///
/// For output layer: δ = (target - output)
/// For hidden layer: δ = Σ(next_weights[i][j] * next_deltas[j]) * output[i] * (1 - output[i])
fn calc_deltas_impl(
    layer: &mut Layer,
    targets: Option<&[f32]>,
    next_layer_deltas: Option<&[f32]>,
    next_layer_weights: Option<&ndarray::Array2<f32>>,
) -> Result<()> {
    // Compute deltas with immutable borrows, then write
    let new_deltas: Vec<f32> = if let Some(targets) = targets {
        // Output layer: δ = (target - output)
        let outputs = layer.outputs();
        if targets.len() != outputs.len() {
            return Err(NeuralNetError::DimensionMismatch {
                expected: outputs.len(),
                actual: targets.len(),
            });
        }

        targets
            .iter()
            .zip(outputs.iter())
            .map(|(&target, &output)| target - output)
            .collect()
    } else if let (Some(next_deltas), Some(next_weights)) = (next_layer_deltas, next_layer_weights) {
        // Hidden layer: δ = Σ(next_weights[i][j] * next_deltas[j]) * output[i] * (1 - output[i])
        let outputs = layer.outputs();
        let num_neurons = layer.num_neurons();

        if next_weights.shape()[0] != num_neurons {
            return Err(NeuralNetError::DimensionMismatch {
                expected: num_neurons,
                actual: next_weights.shape()[0],
            });
        }

        (0..num_neurons)
            .map(|i| {
                // Sum weighted deltas from next layer
                let delta_sum: f32 = (0..next_deltas.len())
                    .map(|j| next_weights[[i, j]] * next_deltas[j])
                    .sum();

                // Multiply by sigmoid derivative: output * (1 - output)
                delta_sum * outputs[i] * (1.0 - outputs[i])
            })
            .collect()
    } else {
        return Err(NeuralNetError::InvalidConfig(
            "calc_deltas requires either targets (output layer) or next layer info (hidden layer)"
                .to_string(),
        ));
    };

    // Write deltas with single mutable borrow
    *layer.deltas_mut() = new_deltas;

    Ok(())
}

/// Update weights using gradient descent
///
/// For each weight: weights[i][j] += learning_rate * deltas[j] * prev_outputs[i]
fn update_weights_impl(
    layer: &mut Layer,
    prev_outputs: &[f32],
    learning_rate: f32,
) -> Result<()> {
    // Cache deltas before taking mutable borrow of weights
    let deltas = layer.deltas().to_vec();

    let weights = layer.weights_mut().ok_or_else(|| {
        NeuralNetError::InvalidConfig("No weights for non-input layer".to_string())
    })?;

    if weights.shape()[0] != prev_outputs.len() {
        return Err(NeuralNetError::DimensionMismatch {
            expected: weights.shape()[0],
            actual: prev_outputs.len(),
        });
    }

    if weights.shape()[1] != deltas.len() {
        return Err(NeuralNetError::DimensionMismatch {
            expected: weights.shape()[1],
            actual: deltas.len(),
        });
    }

    // Update each weight: w[i][j] += η * δ[j] * prev_output[i]
    for i in 0..weights.shape()[0] {
        for j in 0..weights.shape()[1] {
            weights[[i, j]] += learning_rate * deltas[j] * prev_outputs[i];
        }
    }

    Ok(())
}

/// Extension trait for FeedForwardNetwork training
pub trait NetworkTraining {
    /// Train the network for a fixed number of iterations
    ///
    /// # Arguments
    ///
    /// * `inputs` - Training input vectors
    /// * `targets` - Training target vectors
    /// * `iterations` - Number of training iterations
    /// * `learning_rate` - Learning rate (default 0.01 if not specified)
    fn train_by_iteration(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        iterations: usize,
        learning_rate: Option<f32>,
    ) -> Result<()>;

    /// Train the network until error falls below threshold
    ///
    /// # Arguments
    ///
    /// * `inputs` - Training input vectors
    /// * `targets` - Training target vectors
    /// * `target_error` - Stop when MSE falls below this value
    /// * `learning_rate` - Learning rate (default 0.01 if not specified)
    /// * `max_iterations` - Maximum iterations to prevent infinite loops
    ///
    /// # Returns
    ///
    /// The number of iterations it took to reach the target error
    fn train_by_error(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        target_error: f32,
        learning_rate: Option<f32>,
        max_iterations: Option<usize>,
    ) -> Result<usize>;
}

impl NetworkTraining for neural_net_types::FeedForwardNetwork {
    fn train_by_iteration(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        iterations: usize,
        learning_rate: Option<f32>,
    ) -> Result<()> {
        let learning_rate = learning_rate.unwrap_or(0.01);

        for _ in 0..iterations {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                train_single_example(self, input, target, learning_rate)?;
            }
        }

        println!("Training complete after {} iterations", iterations);
        Ok(())
    }

    fn train_by_error(
        &mut self,
        inputs: &[Vec<f32>],
        targets: &[Vec<f32>],
        target_error: f32,
        learning_rate: Option<f32>,
        max_iterations: Option<usize>,
    ) -> Result<usize> {
        use crate::forward::ForwardPropagation;

        let learning_rate = learning_rate.unwrap_or(0.01);
        let max_iterations = max_iterations.unwrap_or(100000);
        let mut error = f32::MAX;
        let mut iteration = 0;

        while error > target_error && iteration < max_iterations {
            error = 0.0;

            for (input, target) in inputs.iter().zip(targets.iter()) {
                // Forward pass to calculate error
                let outputs = self.forward(input)?;

                // Calculate mean squared error
                for (i, &t) in target.iter().enumerate() {
                    let diff = t - outputs[i];
                    error += diff * diff;
                }

                // Backward pass
                train_single_example(self, input, target, learning_rate)?;
            }

            iteration += 1;

            // Print progress every 1000 iterations
            if iteration % 1000 == 0 {
                println!("Iteration {}: error = {:.6}", iteration, error);
            }
        }

        println!(
            "Training complete after {} iterations (final error: {:.6})",
            iteration, error
        );
        Ok(iteration)
    }
}

/// Train the network on a single example (forward + backward pass)
fn train_single_example(
    network: &mut neural_net_types::FeedForwardNetwork,
    input: &[f32],
    target: &[f32],
    learning_rate: f32,
) -> Result<()> {
    use crate::forward::ForwardPropagation;

    // Forward pass
    network.forward(input)?;

    let layer_count = network.layer_count();

    // Backward pass (iterate layers in reverse)
    for i in (1..layer_count).rev() {
        let is_output_layer = i == layer_count - 1;

        if is_output_layer {
            // Output layer: calculate deltas from targets
            network
                .layer_mut(i)
                .ok_or_else(|| NeuralNetError::InvalidConfig("Layer not found".to_string()))?
                .calc_deltas(Some(target), None, None)?;
        } else {
            // Hidden layer: calculate deltas from next layer
            // We need to get next layer's deltas and weights without holding mutable borrow
            let (next_deltas, next_weights) = {
                let next_layer = network.layer(i + 1).ok_or_else(|| {
                    NeuralNetError::InvalidConfig("Next layer not found".to_string())
                })?;
                (next_layer.deltas().to_vec(), next_layer.weights().ok_or_else(|| {
                    NeuralNetError::InvalidConfig("Next layer has no weights".to_string())
                })?.clone())
            };

            network
                .layer_mut(i)
                .ok_or_else(|| NeuralNetError::InvalidConfig("Layer not found".to_string()))?
                .calc_deltas(None, Some(&next_deltas), Some(&next_weights))?;
        }

        // Update weights
        let prev_outputs = network
            .layer(i - 1)
            .ok_or_else(|| NeuralNetError::InvalidConfig("Previous layer not found".to_string()))?
            .outputs()
            .to_vec();

        network
            .layer_mut(i)
            .ok_or_else(|| NeuralNetError::InvalidConfig("Layer not found".to_string()))?
            .update_weights(&prev_outputs, learning_rate)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::ForwardPropagation;
    use ndarray::Array2;

    #[test]
    fn test_untrained_network_performs_poorly() {
        // Create untrained network
        let mut network = neural_net_types::FeedForwardNetwork::new(2, 4, 1);

        // XOR inputs and targets
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        // Calculate error before training
        let mut error_before = 0.0;
        for (input, target) in inputs.iter().zip(&targets) {
            let output = network.forward(input).unwrap();
            let diff = target[0] - output[0];
            error_before += diff * diff;
        }

        // Untrained network should have significant error
        // (With random weights, error is typically > 1.0)
        assert!(
            error_before > 0.5,
            "Untrained network should have significant error, got {}",
            error_before
        );
    }

    #[test]
    fn test_xor_learning() {
        let mut network = neural_net_types::FeedForwardNetwork::new(2, 4, 1);

        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        // Calculate error before training
        let mut error_before = 0.0;
        for (input, target) in inputs.iter().zip(&targets) {
            let output = network.forward(input).unwrap();
            let diff = target[0] - output[0];
            error_before += diff * diff;
        }

        // Train network
        network
            .train_by_error(&inputs, &targets, 0.01, Some(0.1), Some(10000))
            .unwrap();

        // Calculate error after training
        let mut error_after = 0.0;
        let mut correct = 0;
        for (input, target) in inputs.iter().zip(&targets) {
            let output = network.forward(input).unwrap();
            let diff = target[0] - output[0];
            error_after += diff * diff;

            // Check if prediction is correct (within 0.3 threshold)
            if (output[0] - target[0]).abs() < 0.3 {
                correct += 1;
            }
        }

        println!(
            "XOR Learning: error before = {:.4}, error after = {:.4}",
            error_before, error_after
        );
        println!("Correct predictions: {}/4", correct);

        // After training, error should be significantly reduced
        assert!(
            error_after < error_before,
            "Training should reduce error: before={}, after={}",
            error_before,
            error_after
        );

        // Network should learn XOR reasonably well
        assert!(
            error_after < 0.5,
            "Trained network should have low error, got {}",
            error_after
        );

        // Should get most predictions correct
        assert!(
            correct >= 3,
            "Should get at least 3/4 correct, got {}/4",
            correct
        );
    }

    #[test]
    fn test_and_learning() {
        let mut network = neural_net_types::FeedForwardNetwork::new(2, 4, 1);

        // AND truth table (linearly separable)
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![0.0], vec![0.0], vec![1.0]];

        // Train network (AND is easier than XOR)
        network
            .train_by_error(&inputs, &targets, 0.01, Some(0.1), Some(5000))
            .unwrap();

        // Test all examples
        for (input, target) in inputs.iter().zip(&targets) {
            let output = network.forward(input).unwrap();
            let error = (output[0] - target[0]).abs();
            assert!(
                error < 0.3,
                "AND({}, {}) should be ~{}, got {} (error: {})",
                input[0],
                input[1],
                target[0],
                output[0],
                error
            );
        }
    }

    #[test]
    fn test_or_learning() {
        let mut network = neural_net_types::FeedForwardNetwork::new(2, 4, 1);

        // OR truth table (linearly separable)
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![1.0]];

        // Train network (OR is easier than XOR)
        network
            .train_by_error(&inputs, &targets, 0.01, Some(0.1), Some(5000))
            .unwrap();

        // Test all examples
        for (input, target) in inputs.iter().zip(&targets) {
            let output = network.forward(input).unwrap();
            let error = (output[0] - target[0]).abs();
            assert!(
                error < 0.3,
                "OR({}, {}) should be ~{}, got {} (error: {})",
                input[0],
                input[1],
                target[0],
                output[0],
                error
            );
        }
    }

    #[test]
    fn test_output_layer_deltas() {
        let mut layer = Layer::new(2, 2, Some(3));
        // Simulate outputs from forward pass
        layer.set_outputs(vec![0.8, 0.3]);

        let targets = vec![1.0, 0.0];
        layer
            .calc_deltas(Some(&targets), None, None)
            .unwrap();

        // δ = target - output
        assert_eq!(layer.deltas().len(), 2);
        assert!((layer.deltas()[0] - 0.2).abs() < 1e-6); // 1.0 - 0.8
        assert!((layer.deltas()[1] - (-0.3)).abs() < 1e-6); // 0.0 - 0.3
    }

    #[test]
    fn test_hidden_layer_deltas() {
        let mut layer = Layer::new(1, 3, Some(2));
        // Simulate sigmoid outputs from forward pass
        layer.set_outputs(vec![0.5, 0.7, 0.6]);

        // Simulate next layer (2 neurons)
        let next_weights = Array2::from_shape_vec(
            (3, 2),
            vec![
                0.5, 0.2, // weights from neuron 0 to next layer
                0.3, 0.4, // weights from neuron 1 to next layer
                0.1, 0.6, // weights from neuron 2 to next layer
            ],
        )
        .unwrap();
        let next_deltas = vec![0.1, -0.2];

        layer
            .calc_deltas(None, Some(&next_deltas), Some(&next_weights))
            .unwrap();

        // δ[i] = Σ(w[i][j] * δ_next[j]) * output[i] * (1 - output[i])
        assert_eq!(layer.deltas().len(), 3);

        // For neuron 0: (0.5*0.1 + 0.2*(-0.2)) * 0.5 * (1-0.5)
        //              = (0.05 - 0.04) * 0.25 = 0.01 * 0.25 = 0.0025
        assert!((layer.deltas()[0] - 0.0025).abs() < 1e-6);
    }

    #[test]
    fn test_weight_update() {
        let mut layer = Layer::new(1, 2, Some(2));

        // Set known weights
        layer
            .set_weights(Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap())
            .unwrap();

        // Set deltas
        layer.set_deltas(vec![0.1, -0.2]);

        let prev_outputs = vec![0.5, 0.5];
        let learning_rate = 0.01;

        let initial_weights = layer.weights().unwrap().clone();

        layer
            .update_weights(&prev_outputs, learning_rate)
            .unwrap();

        let updated_weights = layer.weights().unwrap();

        // w[i][j] += η * δ[j] * prev_output[i]
        // w[0][0] = 1.0 + 0.01 * 0.1 * 0.5 = 1.0 + 0.0005 = 1.0005
        assert!((updated_weights[[0, 0]] - 1.0005).abs() < 1e-6);
        // w[0][1] = 2.0 + 0.01 * (-0.2) * 0.5 = 2.0 - 0.001 = 1.999
        assert!((updated_weights[[0, 1]] - 1.999).abs() < 1e-6);

        // Verify weights changed
        assert_ne!(initial_weights, *updated_weights);
    }

    #[test]
    fn test_dimension_mismatch_deltas() {
        let mut layer = Layer::new(2, 2, Some(3));
        layer.set_outputs(vec![0.8, 0.3]);

        let wrong_targets = vec![1.0]; // Wrong size
        let result = layer.calc_deltas(Some(&wrong_targets), None, None);

        assert!(matches!(
            result,
            Err(NeuralNetError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_dimension_mismatch_weights() {
        let mut layer = Layer::new(1, 2, Some(2));
        layer.set_deltas(vec![0.1, -0.2]);

        let wrong_prev_outputs = vec![0.5]; // Wrong size (expected 2)
        let result = layer.update_weights(&wrong_prev_outputs, 0.01);

        assert!(matches!(
            result,
            Err(NeuralNetError::DimensionMismatch { .. })
        ));
    }
}
