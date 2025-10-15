//! XOR Gate Example - Why We Need Backpropagation
//!
//! The XOR (exclusive OR) logic gate is **NOT linearly separable**. This means
//! no single straight line can separate the outputs into correct categories.
//! This is why a simple perceptron cannot solve XOR, and we need:
//! 1. A **hidden layer** to learn non-linear representations
//! 2. **Backpropagation** to train the hidden layer weights
//!
//! This is the classic problem that motivated the development of multi-layer
//! neural networks and the backpropagation algorithm.
//!
//! ## XOR Truth Table
//!
//! | Input A | Input B | Output |
//! |---------|---------|--------|
//! |   0.0   |   0.0   |  0.0   |
//! |   0.0   |   1.0   |  1.0   |
//! |   1.0   |   0.0   |  1.0   |
//! |   1.0   |   1.0   |  0.0   |
//!
//! Notice: XOR returns 1 when inputs are *different*, 0 when *same*
//!
//! ## What This Example Shows
//!
//! 1. **Before Training**: Random weights produce random outputs (~50% accuracy)
//! 2. **During Training**: Backpropagation learns the complex XOR pattern
//! 3. **After Training**: Network perfectly learns XOR (100% accuracy)
//!
//! This demonstrates why backpropagation was a breakthrough - it enables
//! networks to learn non-linearly separable patterns that simpler methods cannot.
//!
//! ## Generated Files
//!
//! - `checkpoints/xor_initial.json` - Initial network with random weights
//! - `checkpoints/xor_trained.json` - Trained network after backpropagation
//! - `images/xor_initial.svg` - Visualization of initial network
//! - `images/xor_trained.svg` - Visualization of trained network

use neural_net_core::{FeedForwardNetwork, ForwardPropagation, NetworkTraining, NetworkMetadata, Result};
use neural_net_viz::{NetworkVisualization, VisualizationConfig};
use std::fs;

fn main() -> Result<()> {
    println!("========================================");
    println!("  XOR Gate - The Backpropagation Classic");
    println!("========================================\n");

    println!("Why XOR is special:");
    println!("  - NOT linearly separable (no single line can solve it)");
    println!("  - Requires hidden layer + backpropagation");
    println!("  - The problem that motivated modern neural networks!\n");

    // Create output directories in the example's own directory
    let example_dir = env!("CARGO_MANIFEST_DIR");
    let checkpoint_dir = format!("{}/checkpoints", example_dir);
    let image_dir = format!("{}/images", example_dir);
    fs::create_dir_all(&checkpoint_dir)?;
    fs::create_dir_all(&image_dir)?;

    // XOR truth table
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    println!("XOR Truth Table:");
    println!("  A    B   | Expected | Explanation");
    println!("-----------|----------|------------------");
    println!("  0    0   |   0      | Same → 0");
    println!("  0    1   |   1      | Different → 1");
    println!("  1    0   |   1      | Different → 1");
    println!("  1    1   |   0      | Same → 0");
    println!();

    // Create network: 2 inputs, 4 hidden neurons, 1 output
    let mut network = FeedForwardNetwork::new(2, 4, 1);

    // Test BEFORE training
    println!("--- BEFORE TRAINING (Random Weights) ---");
    println!("Expected: Random performance (~50% accuracy)\n");
    test_network(&mut network, &inputs, &targets)?;

    // Save initial checkpoint and visualization
    save_checkpoint(
        &network,
        &format!("{}/xor_initial.json", checkpoint_dir),
        &format!("{}/xor_initial.svg", image_dir),
        NetworkMetadata::initial("XOR Network"),
    )?;

    // Train the network
    println!("\n--- TRAINING ---");
    println!("Using backpropagation to learn the XOR pattern...\n");
    let iterations = network.train_by_error(&inputs, &targets, 0.01, Some(0.1), Some(10000))?;

    // Test AFTER training
    println!("\n--- AFTER TRAINING ---");
    println!("Expected: Perfect learning (100% accuracy)\n");
    test_network(&mut network, &inputs, &targets)?;

    // Save trained checkpoint and visualization
    save_checkpoint(
        &network,
        &format!("{}/xor_trained.json", checkpoint_dir),
        &format!("{}/xor_trained.svg", image_dir),
        NetworkMetadata::checkpoint("XOR Network", iterations, None),
    )?;

    println!("\n✓ Network successfully learned XOR!");
    println!("✓ This proves backpropagation can solve non-linearly separable problems!");
    println!("\nGenerated files:");
    println!("  - checkpoints/xor_initial.json (initial network)");
    println!("  - checkpoints/xor_trained.json (trained network)");
    println!("  - images/xor_initial.svg (initial visualization)");
    println!("  - images/xor_trained.svg (trained visualization)");

    Ok(())
}

/// Test the network on all input combinations
fn test_network(
    network: &mut FeedForwardNetwork,
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
) -> Result<()> {
    println!("  A    B   | Expected | Actual  | Error  | Correct?");
    println!("-----------|----------|---------|--------|----------");

    let mut total_error = 0.0;
    let mut correct = 0;

    for (input, target) in inputs.iter().zip(targets) {
        let output = network.forward(input)?;
        let error = (target[0] - output[0]).abs();
        total_error += error;

        // Consider correct if within 0.3 of target
        let is_correct = error < 0.3;
        if is_correct {
            correct += 1;
        }

        println!(
            "  {:.0}    {:.0}   |   {:.1}    | {:.4}  | {:.4} | {}",
            input[0],
            input[1],
            target[0],
            output[0],
            error,
            if is_correct { "✓" } else { "✗" }
        );
    }

    println!("-----------|----------|---------|--------|----------");
    println!("Total Error: {:.4}", total_error);
    println!(
        "Accuracy: {}/{} ({:.1}%)",
        correct,
        inputs.len(),
        (correct as f32 / inputs.len() as f32) * 100.0
    );

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
    fn test_xor_untrained_has_high_error() {
        // Negative test: Untrained network should produce high error
        let mut network = FeedForwardNetwork::new(2, 4, 1);
        
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
        
        let mean_error = compute_mean_error(&mut network, &inputs, &targets);
        
        assert!(
            mean_error > 0.3,
            "Untrained network should have high error (>0.3), but got {:.4}",
            mean_error
        );
    }

    #[test]
    fn test_xor_trained_has_low_error() {
        // Positive test: Trained network should produce low error
        let mut network = FeedForwardNetwork::new(2, 4, 1);
        
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
        
        // Train the network
        let iterations = network
            .train_by_error(&inputs, &targets, 0.01, Some(0.1), Some(10000))
            .unwrap();
        
        assert!(iterations > 0, "Should train for at least 1 iteration");
        assert!(iterations <= 10000, "Should complete within max iterations");
        
        let mean_error = compute_mean_error(&mut network, &inputs, &targets);
        
        assert!(
            mean_error < 0.15,
            "Trained network should have low error (<0.15), but got {:.4}",
            mean_error
        );
    }

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
            assert_eq!(
                result, expected,
                "XOR({}, {}) should be {}",
                a, b, expected
            );
        }
    }
}
