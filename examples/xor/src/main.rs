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

use neural_net_core::{FeedForwardNetwork, ForwardPropagation, NetworkTraining, Result};

fn main() -> Result<()> {
    println!("========================================");
    println!("  XOR Gate - The Backpropagation Classic");
    println!("========================================\n");

    println!("Why XOR is special:");
    println!("  - NOT linearly separable (no single line can solve it)");
    println!("  - Requires hidden layer + backpropagation");
    println!("  - The problem that motivated modern neural networks!\n");

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

    // Train the network
    println!("\n--- TRAINING ---");
    println!("Using backpropagation to learn the XOR pattern...\n");
    network.train_by_error(&inputs, &targets, 0.01, Some(0.1), Some(10000))?;

    // Test AFTER training
    println!("\n--- AFTER TRAINING ---");
    println!("Expected: Perfect learning (100% accuracy)\n");
    test_network(&mut network, &inputs, &targets)?;

    println!("\n✓ Network successfully learned XOR!");
    println!("✓ This proves backpropagation can solve non-linearly separable problems!");

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
