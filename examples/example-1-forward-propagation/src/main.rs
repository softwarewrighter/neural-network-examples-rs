//! Forward Propagation Example - XOR Problem
//!
//! This example demonstrates forward propagation through a neural network using the classic
//! XOR problem. XOR is a non-linearly separable problem that cannot be solved by a simple
//! perceptron, but requires a hidden layer.
//!
//! ## The XOR Problem
//!
//! Truth table:
//! | Input A | Input B | Output |
//! |---------|---------|--------|
//! |   0.0   |   0.0   |  0.0   |
//! |   0.0   |   1.0   |  1.0   |
//! |   1.0   |   0.0   |  1.0   |
//! |   1.0   |   1.0   |  0.0   |
//!
//! ## What This Example Teaches
//!
//! 1. **Network Architecture**: 2 inputs → 4 hidden neurons → 1 output
//! 2. **Random Initialization**: Networks start with random weights
//! 3. **Forward Propagation**: How inputs flow through layers to produce outputs
//! 4. **Manual Tuning Limitations**: Why we need backpropagation (manual tuning is impractical)
//! 5. **Checkpointing**: Saving network state at different stages
//! 6. **Visualization**: SVG diagrams showing network structure and weights
//!
//! ## Checkpoints Generated
//!
//! - `checkpoints/xor_initial.json` - Random weights (untrained)
//! - `checkpoints/xor_manual_attempt1.json` - First manual tuning attempt
//! - `checkpoints/xor_manual_attempt2.json` - Second manual tuning attempt
//!
//! ## Visualizations Generated
//!
//! - `images/xor_initial.svg` - Initial network with random weights
//! - `images/xor_manual_attempt1.svg` - After first manual adjustment
//! - `images/xor_manual_attempt2.svg` - After second manual adjustment

use ndarray::Array2;
use neural_net_core::{FeedForwardNetwork, ForwardPropagation, NetworkMetadata, Result};
use neural_net_viz::{NetworkVisualization, VisualizationConfig};
use std::fs;

fn main() -> Result<()> {
    println!("=== XOR Problem - Forward Propagation Example ===\n");

    // Create output directories in the example's own directory
    let example_dir = env!("CARGO_MANIFEST_DIR");
    let checkpoint_dir = format!("{}/checkpoints", example_dir);
    let image_dir = format!("{}/images", example_dir);
    fs::create_dir_all(&checkpoint_dir)?;
    fs::create_dir_all(&image_dir)?;

    // Define XOR training data
    let xor_inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let xor_targets = vec![0.0, 1.0, 1.0, 0.0];

    println!("XOR Truth Table:");
    for (input, &target) in xor_inputs.iter().zip(&xor_targets) {
        println!("  {} XOR {} = {}", input[0], input[1], target);
    }
    println!();

    // Stage 1: Initial network with random weights
    println!("--- Stage 1: Initial Network (Random Weights) ---");
    let mut network = FeedForwardNetwork::new(2, 4, 1);

    test_network(&mut network, &xor_inputs, &xor_targets, "Initial (Random)")?;

    save_checkpoint(
        &network,
        &format!("{}/xor_initial.json", checkpoint_dir),
        &format!("{}/xor_initial.svg", image_dir),
        NetworkMetadata::initial("XOR Network"),
    )?;

    // Stage 2: Manual weight tuning attempt 1
    // Try to improve the network by manually adjusting weights
    // (This demonstrates why backpropagation is needed!)
    println!("\n--- Stage 2: Manual Weight Tuning Attempt 1 ---");
    println!("Manually adjusting weights (guessing what might work better)...");

    // Access layer 1 (hidden) and manually set some weights
    // This is a crude attempt to show manual tuning is impractical
    network.layer_mut(1).unwrap().set_weights(
        Array2::from_shape_vec(
            (2, 4),
            vec![
                0.8, -0.5, 0.6, -0.3, // weights from input 0
                0.7, -0.6, 0.5, -0.4, // weights from input 1
            ],
        )
        .unwrap(),
    )?;

    test_network(&mut network, &xor_inputs, &xor_targets, "Manual Attempt 1")?;

    save_checkpoint(
        &network,
        &format!("{}/xor_manual_attempt1.json", checkpoint_dir),
        &format!("{}/xor_manual_attempt1.svg", image_dir),
        NetworkMetadata::checkpoint("XOR Network - Manual Tuning Attempt 1", 0, None),
    )?;

    // Stage 3: Manual weight tuning attempt 2
    println!("\n--- Stage 3: Manual Weight Tuning Attempt 2 ---");
    println!("Trying different manual adjustments...");

    // Adjust output layer weights
    network
        .layer_mut(2)
        .unwrap()
        .set_weights(Array2::from_shape_vec((4, 1), vec![1.5, -1.2, 1.3, -1.1]).unwrap())?;

    test_network(&mut network, &xor_inputs, &xor_targets, "Manual Attempt 2")?;

    save_checkpoint(
        &network,
        &format!("{}/xor_manual_attempt2.json", checkpoint_dir),
        &format!("{}/xor_manual_attempt2.svg", image_dir),
        NetworkMetadata::checkpoint("XOR Network - Manual Tuning Attempt 2", 0, None),
    )?;

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("1. Random weights produce random outputs (not useful)");
    println!("2. Manual weight tuning is impractical (too many weights, complex interactions)");
    println!("3. We need an automatic learning algorithm → Backpropagation!");
    println!("\nGenerated files:");
    println!("  - checkpoints/xor_*.json (network state)");
    println!("  - images/xor_*.svg (visualizations)");
    println!("\nNext example will implement backpropagation for automatic learning.");

    Ok(())
}

/// Test the network on XOR inputs and display results
fn test_network(
    network: &mut FeedForwardNetwork,
    inputs: &[Vec<f32>],
    targets: &[f32],
    stage_name: &str,
) -> Result<()> {
    println!("Testing: {}", stage_name);
    println!("  Input    Target  Output   Error");
    println!("  -------  ------  -------  -------");

    let mut total_error = 0.0;

    for (input, &target) in inputs.iter().zip(targets) {
        let output = network.forward(input)?;
        let error = (output[0] - target).abs();
        total_error += error;

        println!(
            "  [{}, {}]   {:.1}     {:.4}   {:.4}",
            input[0], input[1], target, output[0], error
        );
    }

    let mean_error = total_error / inputs.len() as f32;
    println!("  Mean Absolute Error: {:.4}", mean_error);

    Ok(())
}

/// Save network checkpoint and visualization
fn save_checkpoint(
    network: &FeedForwardNetwork,
    checkpoint_path: &str,
    svg_path: &str,
    metadata: NetworkMetadata,
) -> Result<()> {
    // Save JSON checkpoint
    network.save_checkpoint(checkpoint_path, metadata.clone())?;
    println!("  ✓ Saved checkpoint: {}", checkpoint_path);

    // Generate and save SVG visualization
    let config = VisualizationConfig::default();
    network.save_svg_with_metadata(svg_path, &metadata, &config)?;
    println!("  ✓ Saved visualization: {}", svg_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_truth_table() {
        // Verify XOR logic is correct
        let test_cases = vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ];

        for (input, expected) in test_cases {
            let a = input[0] as u8;
            let b = input[1] as u8;
            let result = (a ^ b) as f32;
            assert_eq!(result, expected, "XOR({}, {}) should be {}", a, b, expected);
        }
    }

    #[test]
    fn test_untrained_network_has_high_error() {
        // Negative test: Untrained network with random weights should produce high error
        let mut network = FeedForwardNetwork::new(2, 4, 1);

        let xor_inputs = [
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let xor_targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        let mut total_error = 0.0;
        for (input, target) in xor_inputs.iter().zip(&xor_targets) {
            let output = network.forward(input).unwrap();
            total_error += (output[0] - target[0]).abs();
        }
        let mean_error = total_error / xor_inputs.len() as f32;

        assert!(
            mean_error > 0.3,
            "Untrained network should have high error (>0.3), but got {:.4}",
            mean_error
        );
    }

    #[test]
    fn test_forward_propagation_xor() {
        let mut network = FeedForwardNetwork::new(2, 4, 1);

        // Test that network can process all XOR inputs
        let xor_inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];

        for input in &xor_inputs {
            let output = network.forward(input).unwrap();
            assert_eq!(output.len(), 1, "Should have 1 output");
            // With random weights, just verify we get some output
            assert!(!output[0].is_nan(), "Output should not be NaN");
        }
    }

    #[test]
    fn test_checkpoint_save_load() {
        use std::env;

        let network = FeedForwardNetwork::new(2, 3, 1);
        let metadata = NetworkMetadata::initial("Test Network");
        let temp_path = env::temp_dir().join("test_checkpoint.json");

        // Save checkpoint
        network
            .save_checkpoint(&temp_path, metadata.clone())
            .unwrap();

        // Load checkpoint
        let (loaded_network, loaded_metadata) =
            FeedForwardNetwork::load_checkpoint(&temp_path).unwrap();

        // Verify same architecture
        assert_eq!(loaded_network.layer_count(), network.layer_count());
        assert_eq!(loaded_metadata.name, metadata.name);

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_svg_generation() {
        use std::env;

        let network = FeedForwardNetwork::new(2, 4, 1);
        let config = VisualizationConfig::default();
        let temp_path = env::temp_dir().join("test_network.svg");

        // Save SVG
        network.save_svg(&temp_path, &config).unwrap();

        // Verify file exists and contains SVG
        let content = std::fs::read_to_string(&temp_path).unwrap();
        assert!(content.contains("<svg"));
        assert!(content.contains("neurons"));
        assert!(content.contains("connections"));

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }
}
