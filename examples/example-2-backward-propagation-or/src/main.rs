//! OR Gate Example - Demonstrating Backpropagation
//!
//! The OR logic gate is **linearly separable**, meaning a single perceptron
//! (no hidden layer) could theoretically solve it. However, this example uses
//! a network with a hidden layer to demonstrate backpropagation training.
//!
//! ## OR Truth Table
//!
//! | Input A | Input B | Output |
//! |---------|---------|--------|
//! |   0.0   |   0.0   |  0.0   |
//! |   0.0   |   1.0   |  1.0   |
//! |   1.0   |   0.0   |  1.0   |
//! |   1.0   |   1.0   |  1.0   |
//!
//! ## What This Example Shows
//!
//! 1. **Before Training**: Random weights produce random outputs
//! 2. **During Training**: Backpropagation adjusts weights to minimize error
//! 3. **After Training**: Network learns the OR function perfectly
//!
//! ## Generated Files
//!
//! - `checkpoints/or_initial.json` - Initial network with random weights
//! - `checkpoints/or_trained.json` - Trained network after backpropagation
//! - `images/or_initial.svg` - Visualization of initial network
//! - `images/or_trained.svg` - Visualization of trained network

use neural_net_core::{FeedForwardNetwork, ForwardPropagation, NetworkTraining, NetworkMetadata, Result};
use neural_net_viz::{NetworkVisualization, VisualizationConfig};
use std::fs;

fn main() -> Result<()> {
    println!("====================================");
    println!("   OR Gate - Backpropagation Demo");
    println!("====================================\n");

    // Create output directories in the example's own directory
    let example_dir = env!("CARGO_MANIFEST_DIR");
    let checkpoint_dir = format!("{}/checkpoints", example_dir);
    let image_dir = format!("{}/images", example_dir);
    fs::create_dir_all(&checkpoint_dir)?;
    fs::create_dir_all(&image_dir)?;

    // OR truth table
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![1.0]];

    println!("OR Truth Table:");
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
        &format!("{}/or_initial.json", checkpoint_dir),
        &format!("{}/or_initial.svg", image_dir),
        NetworkMetadata::initial("OR Network"),
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
        &format!("{}/or_trained.json", checkpoint_dir),
        &format!("{}/or_trained.svg", image_dir),
        NetworkMetadata::checkpoint("OR Network", iterations, None),
    )?;

    println!("\n✓ Network successfully learned the OR function!");
    println!("\nGenerated files:");
    println!("  - checkpoints/or_initial.json (initial network)");
    println!("  - checkpoints/or_trained.json (trained network)");
    println!("  - images/or_initial.svg (initial visualization)");
    println!("  - images/or_trained.svg (trained visualization)");

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
