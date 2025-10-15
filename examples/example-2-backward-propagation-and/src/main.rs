//! AND Gate Example - Demonstrating Backpropagation
//!
//! The AND logic gate is **linearly separable**, meaning a single perceptron
//! (no hidden layer) could theoretically solve it. However, this example uses
//! a network with a hidden layer to demonstrate backpropagation training.
//!
//! ## AND Truth Table
//!
//! | Input A | Input B | Output |
//! |---------|---------|--------|
//! |   0.0   |   0.0   |  0.0   |
//! |   0.0   |   1.0   |  0.0   |
//! |   1.0   |   0.0   |  0.0   |
//! |   1.0   |   1.0   |  1.0   |
//!
//! ## What This Example Shows
//!
//! 1. **Before Training**: Random weights produce random outputs
//! 2. **During Training**: Backpropagation adjusts weights to minimize error
//! 3. **After Training**: Network learns the AND function perfectly
//!
//! ## Generated Files
//!
//! - `checkpoints/and_initial.json` - Initial network with random weights
//! - `checkpoints/and_trained.json` - Trained network after backpropagation
//! - `images/and_initial.svg` - Visualization of initial network
//! - `images/and_trained.svg` - Visualization of trained network

use neural_net_core::{FeedForwardNetwork, ForwardPropagation, NetworkTraining, NetworkMetadata, Result};
use neural_net_viz::{NetworkVisualization, VisualizationConfig};
use std::fs;

fn main() -> Result<()> {
    println!("====================================");
    println!("   AND Gate - Backpropagation Demo");
    println!("====================================\n");

    // Create output directories in the example's own directory
    let example_dir = env!("CARGO_MANIFEST_DIR");
    let checkpoint_dir = format!("{}/checkpoints", example_dir);
    let image_dir = format!("{}/images", example_dir);
    fs::create_dir_all(&checkpoint_dir)?;
    fs::create_dir_all(&image_dir)?;

    // AND truth table
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![0.0], vec![0.0], vec![1.0]];

    println!("AND Truth Table:");
    println!("  A    B   | Expected");
    println!("-----------|----------");
    for (input, target) in inputs.iter().zip(&targets) {
        println!("  {:.0}    {:.0}   |   {:.0}", input[0], input[1], target[0]);
    }
    println!();

    // Create network: 2 inputs, 4 hidden neurons, 1 output
    let mut network = FeedForwardNetwork::new(2, 4, 1);

    // Test BEFORE training
    println!("--- BEFORE TRAINING (Random Weights) ---");
    test_network(&mut network, &inputs, &targets)?;

    // Save initial checkpoint and visualization
    save_checkpoint(
        &network,
        &format!("{}/and_initial.json", checkpoint_dir),
        &format!("{}/and_initial.svg", image_dir),
        NetworkMetadata::initial("AND Network"),
    )?;

    // Train the network
    println!("\n--- TRAINING ---");
    let iterations = network.train_by_error(&inputs, &targets, 0.01, Some(0.1), Some(5000))?;

    // Test AFTER training
    println!("\n--- AFTER TRAINING ---");
    test_network(&mut network, &inputs, &targets)?;

    // Save trained checkpoint and visualization
    save_checkpoint(
        &network,
        &format!("{}/and_trained.json", checkpoint_dir),
        &format!("{}/and_trained.svg", image_dir),
        NetworkMetadata::checkpoint("AND Network", iterations, None),
    )?;

    println!("\n✓ Network successfully learned the AND function!");
    println!("\nGenerated files:");
    println!("  - checkpoints/and_initial.json (initial network)");
    println!("  - checkpoints/and_trained.json (trained network)");
    println!("  - images/and_initial.svg (initial visualization)");
    println!("  - images/and_trained.svg (trained visualization)");

    Ok(())
}

/// Test the network on all input combinations
fn test_network(
    network: &mut FeedForwardNetwork,
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
) -> Result<()> {
    println!("  A    B   | Expected | Actual  | Error");
    println!("-----------|----------|---------|-------");

    let mut total_error = 0.0;
    let mut correct = 0;

    for (input, target) in inputs.iter().zip(targets) {
        let output = network.forward(input)?;
        let error = (target[0] - output[0]).abs();
        total_error += error;

        // Consider correct if within 0.3 of target
        if error < 0.3 {
            correct += 1;
        }

        println!(
            "  {:.0}    {:.0}   |   {:.1}    | {:.4}  | {:.4}",
            input[0], input[1], target[0], output[0], error
        );
    }

    println!("-----------|----------|---------|-------");
    println!("Total Error: {:.4}", total_error);
    println!("Correct: {}/{}", correct, inputs.len());

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

    /// Helper function to compute mean absolute error
    fn compute_mean_error(network: &mut FeedForwardNetwork, inputs: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
        let mut total_error = 0.0;
        for (input, target) in inputs.iter().zip(targets) {
            let output = network.forward(input).unwrap();
            total_error += (output[0] - target[0]).abs();
        }
        total_error / inputs.len() as f32
    }

    #[test]
    fn test_and_untrained_has_high_error() {
        // Negative test: Untrained network should produce high error
        let mut network = FeedForwardNetwork::new(2, 4, 1);
        
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![0.0], vec![0.0], vec![1.0]];
        
        let mean_error = compute_mean_error(&mut network, &inputs, &targets);
        
        assert!(
            mean_error > 0.3,
            "Untrained network should have high error (>0.3), but got {:.4}",
            mean_error
        );
    }

    #[test]
    fn test_and_trained_has_low_error() {
        // Positive test: Trained network should produce low error
        let mut network = FeedForwardNetwork::new(2, 4, 1);
        
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![0.0], vec![0.0], vec![1.0]];
        
        // Train the network
        let iterations = network
            .train_by_error(&inputs, &targets, 0.01, Some(0.1), Some(5000))
            .unwrap();
        
        assert!(iterations > 0, "Should train for at least 1 iteration");
        assert!(iterations <= 5000, "Should complete within max iterations");
        
        let mean_error = compute_mean_error(&mut network, &inputs, &targets);
        
        assert!(
            mean_error < 0.15,
            "Trained network should have low error (<0.15), but got {:.4}",
            mean_error
        );
    }

    #[test]
    fn test_and_truth_table() {
        // Verify AND logic is correct
        let test_cases = vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 0.0),
            (vec![1.0, 0.0], 0.0),
            (vec![1.0, 1.0], 1.0),
        ];
        
        for (input, expected) in test_cases {
            let a = input[0] as u8;
            let b = input[1] as u8;
            let result = (a & b) as f32;
            assert_eq!(
                result, expected,
                "AND({}, {}) should be {}",
                a, b, expected
            );
        }
    }
}
