//! 3-Input Majority Example - Voting Function
//!
//! This example demonstrates learning the majority function: output 1 if at least
//! 2 of the 3 inputs are 1. This is similar to a voting system (2 out of 3).
//!
//! ## The 3-Input Majority Problem
//!
//! Truth table:
//! | A | B | C | Majority (≥2 ones) |
//! |---|---|---|---------------------|
//! | 0 | 0 | 0 |         0           |
//! | 0 | 0 | 1 |         0           |
//! | 0 | 1 | 0 |         0           |
//! | 0 | 1 | 1 |         1           |
//! | 1 | 0 | 0 |         0           |
//! | 1 | 0 | 1 |         1           |
//! | 1 | 1 | 0 |         1           |
//! | 1 | 1 | 1 |         1           |
//!
//! ## What This Example Teaches
//!
//! 1. **Voting Logic**: Neural networks can learn democratic decision-making
//! 2. **Threshold Functions**: Majority is a natural threshold (≥2 out of 3)
//! 3. **Linear Separability**: Unlike XOR/parity, majority IS linearly separable
//! 4. **Faster Convergence**: Linearly separable functions train faster
//! 5. **Boolean Algebra**: Majority(A,B,C) = (A·B) + (A·C) + (B·C)
//!
//! ## Network Architecture
//!
//! - **Input Layer**: 3 neurons (A, B, C)
//! - **Hidden Layer**: 4 neurons (fewer than parity - simpler function)
//! - **Output Layer**: 1 neuron (majority decision)
//!
//! ## Expected Training Time
//!
//! - Iterations: ~1000-2000 (faster than parity!)
//! - Learning rate: 0.5 (standard)
//! - Target error: <0.1

use neural_net_core::{
    FeedForwardNetwork, ForwardPropagation, NetworkMetadata, NetworkTraining, Result,
};
use neural_net_viz::{NetworkVisualization, VisualizationConfig};
use std::fs;

fn main() -> Result<()> {
    println!("=== 3-Input Majority Problem ===\n");

    // Create output directories in the example's own directory
    let example_dir = env!("CARGO_MANIFEST_DIR");
    let checkpoint_dir = format!("{}/checkpoints", example_dir);
    let image_dir = format!("{}/images", example_dir);
    fs::create_dir_all(&checkpoint_dir)?;
    fs::create_dir_all(&image_dir)?;

    // Define majority function training data
    let majority_inputs = vec![
        vec![0.0, 0.0, 0.0], // 0 ones -> majority 0
        vec![0.0, 0.0, 1.0], // 1 one  -> majority 0
        vec![0.0, 1.0, 0.0], // 1 one  -> majority 0
        vec![0.0, 1.0, 1.0], // 2 ones -> majority 1
        vec![1.0, 0.0, 0.0], // 1 one  -> majority 0
        vec![1.0, 0.0, 1.0], // 2 ones -> majority 1
        vec![1.0, 1.0, 0.0], // 2 ones -> majority 1
        vec![1.0, 1.0, 1.0], // 3 ones -> majority 1
    ];
    let majority_targets = vec![
        vec![0.0],
        vec![0.0],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![1.0],
    ];

    println!("3-Input Majority Truth Table:");
    println!("  A  B  C | Majority");
    println!("  --------|----------");
    for (input, target) in majority_inputs.iter().zip(&majority_targets) {
        println!(
            "  {}  {}  {} |    {}",
            input[0] as u8, input[1] as u8, input[2] as u8, target[0] as u8
        );
    }
    println!();

    // Create network: 3 inputs → 4 hidden → 1 output
    println!("Creating network: 3 inputs → 4 hidden → 1 output\n");
    let mut network = FeedForwardNetwork::new(3, 4, 1);

    // Save initial state
    println!("--- Initial Network (Random Weights) ---");
    test_network(&mut network, &majority_inputs, &majority_targets, "Initial")?;
    save_checkpoint(
        &network,
        &format!("{}/majority3_initial.json", checkpoint_dir),
        &format!("{}/majority3_initial.svg", image_dir),
        NetworkMetadata::initial("3-Input Majority Network"),
    )?;

    // Train the network
    println!("\n--- Training ---");
    println!("Learning rate: 0.5");
    println!("Target error: 0.1");
    println!("Max iterations: 5000\n");

    let iterations = network.train_by_error(
        &majority_inputs,
        &majority_targets,
        0.5,
        Some(0.1),
        Some(5000),
    )?;

    println!("\nTraining completed in {} iterations", iterations);

    // Test trained network
    println!("\n--- Trained Network ---");
    test_network(&mut network, &majority_inputs, &majority_targets, "Trained")?;
    save_checkpoint(
        &network,
        &format!("{}/majority3_trained.json", checkpoint_dir),
        &format!("{}/majority3_trained.svg", image_dir),
        NetworkMetadata::checkpoint("3-Input Majority Network - Trained", iterations, None),
    )?;

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("1. Majority function is linearly separable (unlike XOR/parity)");
    println!("2. Trains faster than parity (~1000-2000 vs ~3000-5000 iterations)");
    println!("3. Represents voting logic: output 1 if ≥2 inputs are 1");
    println!("4. Boolean algebra: Majority = (A·B) + (A·C) + (B·C)");
    println!("5. Real-world analogy: 2-out-of-3 voting systems");
    println!("\nGenerated files:");
    println!("  - checkpoints/majority3_*.json (network state)");
    println!("  - images/majority3_*.svg (visualizations)");

    Ok(())
}

/// Test the network on majority inputs and display results
fn test_network(
    network: &mut FeedForwardNetwork,
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
    stage_name: &str,
) -> Result<()> {
    println!("Testing: {}", stage_name);
    println!("  Input      Target  Output   Error");
    println!("  ---------- ------  -------  -------");

    let mut total_error = 0.0;

    for (input, target) in inputs.iter().zip(targets) {
        let output = network.forward(input)?;
        let error = (output[0] - target[0]).abs();
        total_error += error;

        println!(
            "  {} {} {}    {:.1}     {:.4}   {:.4}",
            input[0] as u8, input[1] as u8, input[2] as u8, target[0], output[0], error
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
    fn test_majority3_truth_table() {
        // Verify our truth table is correct
        let inputs = [
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ];
        let expected = vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0];

        for (input, &expected_output) in inputs.iter().zip(&expected) {
            let count = input.iter().filter(|&&x| x == 1.0).count();
            let majority = if count >= 2 { 1.0 } else { 0.0 };
            assert_eq!(
                majority, expected_output,
                "Majority of {:?} should be {}",
                input, expected_output
            );
        }
    }

    #[test]
    fn test_majority3_untrained_has_high_error() {
        // Negative test: Untrained network should produce high error
        let mut network = FeedForwardNetwork::new(3, 4, 1);

        let inputs = [
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ];
        let targets = vec![
            vec![0.0],
            vec![0.0],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
        ];

        let mut total_error = 0.0;
        for (input, target) in inputs.iter().zip(&targets) {
            let output = network.forward(input).unwrap();
            total_error += (output[0] - target[0]).abs();
        }
        let mean_error = total_error / inputs.len() as f32;

        assert!(
            mean_error > 0.3,
            "Untrained network should have high error (>0.3), but got {:.4}",
            mean_error
        );
    }

    #[test]
    fn test_majority3_network_trains() {
        let inputs = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ];
        let targets = vec![
            vec![0.0],
            vec![0.0],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![1.0],
        ];

        let mut network = FeedForwardNetwork::new(3, 4, 1);
        let iterations = network
            .train_by_error(&inputs, &targets, 0.5, Some(0.1), Some(5000))
            .unwrap();

        assert!(iterations > 0, "Should train for at least 1 iteration");
        assert!(iterations <= 5000, "Should complete within max iterations");

        // Verify trained network is accurate
        for (input, target) in inputs.iter().zip(&targets) {
            let output = network.forward(input).unwrap();
            let error = (output[0] - target[0]).abs();
            assert!(
                error < 0.5,
                "Output {} should be close to target {} (error: {})",
                output[0],
                target[0],
                error
            );
        }
    }
}
