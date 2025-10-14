//! Forward Propagation Example
//!
//! This example demonstrates forward propagation through a simple 3-layer neural network.
//! We manually create layers, set known weights, and propagate inputs through the network
//! to understand how information flows forward.

use neural_net_core::{Layer, Result};
use ndarray::Array2;

fn main() -> Result<()> {
    println!("=== Forward Propagation Example ===\n");

    // Create a simple 3-layer network: 2 inputs -> 3 hidden -> 2 outputs
    println!("Network Architecture:");
    println!("  Input layer:  2 neurons");
    println!("  Hidden layer: 3 neurons (sigmoid activation)");
    println!("  Output layer: 2 neurons (linear activation)\n");

    // Create layers
    let mut input_layer = Layer::new(0, 2, None);
    let mut hidden_layer = Layer::new(1, 3, Some(2));
    let mut output_layer = Layer::new(2, 2, Some(3));

    // Set known weights for reproducible results
    // Hidden layer weights: [2 inputs x 3 hidden neurons]
    hidden_layer.set_weights(Array2::from_shape_vec(
        (2, 3),
        vec![
            0.5, 0.3, 0.2,  // weights from input neuron 0
            0.4, 0.6, 0.1,  // weights from input neuron 1
        ],
    ).unwrap())?;

    // Output layer weights: [3 hidden x 2 output neurons]
    output_layer.set_weights(Array2::from_shape_vec(
        (3, 2),
        vec![
            0.7, 0.2,  // weights from hidden neuron 0
            0.5, 0.4,  // weights from hidden neuron 1
            0.3, 0.6,  // weights from hidden neuron 2
        ],
    ).unwrap())?;

    // Example 1: Forward propagate with input [1.0, 0.0]
    println!("--- Example 1: Input [1.0, 0.0] ---");
    forward_propagate_example(&mut input_layer, &mut hidden_layer, &mut output_layer, vec![1.0, 0.0])?;

    // Example 2: Forward propagate with input [0.5, 0.8]
    println!("\n--- Example 2: Input [0.5, 0.8] ---");
    forward_propagate_example(&mut input_layer, &mut hidden_layer, &mut output_layer, vec![0.5, 0.8])?;

    // Example 3: Forward propagate with input [0.0, 1.0]
    println!("\n--- Example 3: Input [0.0, 1.0] ---");
    forward_propagate_example(&mut input_layer, &mut hidden_layer, &mut output_layer, vec![0.0, 1.0])?;

    println!("\n=== Forward Propagation Complete ===");

    Ok(())
}

/// Perform forward propagation through the network with given inputs
fn forward_propagate_example(
    input_layer: &mut Layer,
    hidden_layer: &mut Layer,
    output_layer: &mut Layer,
    inputs: Vec<f32>,
) -> Result<()> {
    println!("Input: {:?}", inputs);

    // Set inputs to input layer
    input_layer.set_inputs(inputs.clone());
    input_layer.forward_propagate(None, false)?;
    println!("Input layer outputs: {:?}", input_layer.outputs());

    // Propagate through hidden layer
    hidden_layer.forward_propagate(Some(input_layer.outputs()), false)?;
    println!("Hidden layer inputs:  {:?}",
        hidden_layer.inputs().iter().map(|&x| format!("{:.4}", x)).collect::<Vec<_>>());
    println!("Hidden layer outputs: {:?}",
        hidden_layer.outputs().iter().map(|&x| format!("{:.4}", x)).collect::<Vec<_>>());

    // Propagate through output layer
    output_layer.forward_propagate(Some(hidden_layer.outputs()), true)?;
    println!("Output layer inputs:  {:?}",
        output_layer.inputs().iter().map(|&x| format!("{:.4}", x)).collect::<Vec<_>>());
    println!("Output layer outputs: {:?}",
        output_layer.outputs().iter().map(|&x| format!("{:.4}", x)).collect::<Vec<_>>());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_propagation_specific_input() {
        // Create layers
        let mut input_layer = Layer::new(0, 2, None);
        let mut hidden_layer = Layer::new(1, 3, Some(2));
        let mut output_layer = Layer::new(2, 2, Some(3));

        // Set known weights
        hidden_layer.set_weights(Array2::from_shape_vec(
            (2, 3),
            vec![0.5, 0.3, 0.2, 0.4, 0.6, 0.1],
        ).unwrap()).unwrap();

        output_layer.set_weights(Array2::from_shape_vec(
            (3, 2),
            vec![0.7, 0.2, 0.5, 0.4, 0.3, 0.6],
        ).unwrap()).unwrap();

        // Test with input [1.0, 0.0]
        input_layer.set_inputs(vec![1.0, 0.0]);
        input_layer.forward_propagate(None, false).unwrap();

        hidden_layer.forward_propagate(Some(input_layer.outputs()), false).unwrap();

        // Expected hidden inputs: [1.0*0.5 + 0.0*0.4, 1.0*0.3 + 0.0*0.6, 1.0*0.2 + 0.0*0.1]
        //                        = [0.5, 0.3, 0.2]
        assert!((hidden_layer.inputs()[0] - 0.5).abs() < 1e-6);
        assert!((hidden_layer.inputs()[1] - 0.3).abs() < 1e-6);
        assert!((hidden_layer.inputs()[2] - 0.2).abs() < 1e-6);

        // Hidden outputs should be sigmoid([0.5, 0.3, 0.2])
        // sigmoid(0.5) ≈ 0.6225, sigmoid(0.3) ≈ 0.5744, sigmoid(0.2) ≈ 0.5498
        assert!((hidden_layer.outputs()[0] - 0.6225).abs() < 1e-3);
        assert!((hidden_layer.outputs()[1] - 0.5744).abs() < 1e-3);
        assert!((hidden_layer.outputs()[2] - 0.5498).abs() < 1e-3);

        output_layer.forward_propagate(Some(hidden_layer.outputs()), true).unwrap();

        // Output should have 2 values
        assert_eq!(output_layer.outputs().len(), 2);
    }

    #[test]
    fn test_multiple_forward_passes() {
        let mut input_layer = Layer::new(0, 2, None);
        let mut hidden_layer = Layer::new(1, 2, Some(2));
        let mut output_layer = Layer::new(2, 1, Some(2));

        // Set simple weights for testing
        hidden_layer.set_weights(Array2::from_shape_vec(
            (2, 2),
            vec![2.0, 0.5, 0.5, 2.0], // Non-identity weights for more variation
        ).unwrap()).unwrap();

        output_layer.set_weights(Array2::from_shape_vec(
            (2, 1),
            vec![1.0, -1.0], // Difference between hidden outputs
        ).unwrap()).unwrap();

        // First pass with [1.0, 0.0]
        input_layer.set_inputs(vec![1.0, 0.0]);
        input_layer.forward_propagate(None, false).unwrap();
        hidden_layer.forward_propagate(Some(input_layer.outputs()), false).unwrap();
        output_layer.forward_propagate(Some(hidden_layer.outputs()), true).unwrap();

        let output1 = output_layer.outputs()[0];

        // Second pass with [0.0, 1.0] (opposite input)
        input_layer.set_inputs(vec![0.0, 1.0]);
        input_layer.forward_propagate(None, false).unwrap();
        hidden_layer.forward_propagate(Some(input_layer.outputs()), false).unwrap();
        output_layer.forward_propagate(Some(hidden_layer.outputs()), true).unwrap();

        let output2 = output_layer.outputs()[0];

        // Outputs should be different for different inputs
        assert!((output1 - output2).abs() > 0.01,
            "Expected significant difference between outputs, got |{} - {}| = {}",
            output1, output2, (output1 - output2).abs());
    }

    #[test]
    fn test_linear_activation_output_layer() {
        // Create an input layer and output layer to test linear activation
        let mut input_layer = Layer::new(0, 3, None);
        let mut output_layer = Layer::new(2, 2, Some(3));

        // Set inputs on input layer
        input_layer.set_inputs(vec![0.5, -0.3, 0.8]);
        input_layer.forward_propagate(None, false).unwrap();

        output_layer.set_weights(Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ).unwrap()).unwrap();

        output_layer.forward_propagate(Some(input_layer.outputs()), true).unwrap();

        // Output layer should use linear activation
        // So outputs == inputs (no sigmoid transformation)
        assert_eq!(output_layer.outputs(), output_layer.inputs());
    }
}
