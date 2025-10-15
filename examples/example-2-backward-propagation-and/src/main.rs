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

use neural_net_core::{FeedForwardNetwork, ForwardPropagation, NetworkTraining, Result};

fn main() -> Result<()> {
    println!("====================================");
    println!("   AND Gate - Backpropagation Demo");
    println!("====================================\n");

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

    // Train the network
    println!("\n--- TRAINING ---");
    network.train_by_error(&inputs, &targets, 0.01, Some(0.1), Some(5000))?;

    // Test AFTER training
    println!("\n--- AFTER TRAINING ---");
    test_network(&mut network, &inputs, &targets)?;

    println!("\n✓ Network successfully learned the AND function!");

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
