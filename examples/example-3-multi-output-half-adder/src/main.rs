//! Half Adder Example - Multi-Output Network
//!
//! This example demonstrates learning a half adder circuit with multiple outputs.
//! A half adder adds two binary digits and produces a sum and carry output.
//!
//! ## The Half Adder Problem
//!
//! Truth table:
//! | A | B | Sum | Carry |
//! |---|---|-----|-------|
//! | 0 | 0 |  0  |   0   |
//! | 0 | 1 |  1  |   0   |
//! | 1 | 0 |  1  |   0   |
//! | 1 | 1 |  0  |   1   |
//!
//! ## What This Example Teaches
//!
//! 1. **Multi-Output Networks**: Networks can learn multiple functions simultaneously
//! 2. **Digital Logic**: Half adder is a fundamental digital circuit component
//! 3. **Related Outputs**: Sum = XOR(A,B), Carry = AND(A,B)
//! 4. **Shared Representations**: Hidden layer learns features useful for both outputs
//! 5. **Training Complexity**: Multiple outputs may need more hidden neurons
//!
//! ## Network Architecture
//!
//! - **Input Layer**: 2 neurons (A, B)
//! - **Hidden Layer**: 4 neurons (learn shared features for sum and carry)
//! - **Output Layer**: 2 neurons (sum, carry)
//!
//! ## Expected Training Time
//!
//! - Iterations: ~1500-2500
//! - Learning rate: 0.5 (standard)
//! - Target error: <0.1

use neural_net_core::{FeedForwardNetwork, ForwardPropagation, NetworkTraining, NetworkMetadata, Result};
use neural_net_viz::{NetworkVisualization, VisualizationConfig};
use std::fs;

fn main() -> Result<()> {
    println!("=== Half Adder Problem ===\n");

    // Create output directories in the example's own directory
    let example_dir = env!("CARGO_MANIFEST_DIR");
    let checkpoint_dir = format!("{}/checkpoints", example_dir);
    let image_dir = format!("{}/images", example_dir);
    fs::create_dir_all(&checkpoint_dir)?;
    fs::create_dir_all(&image_dir)?;

    // Define half adder training data
    let adder_inputs = vec![
        vec![0.0, 0.0], // 0 + 0
        vec![0.0, 1.0], // 0 + 1
        vec![1.0, 0.0], // 1 + 0
        vec![1.0, 1.0], // 1 + 1
    ];
    // Each target has two outputs: [sum, carry]
    let adder_targets = vec![
        vec![0.0, 0.0], // sum=0, carry=0
        vec![1.0, 0.0], // sum=1, carry=0
        vec![1.0, 0.0], // sum=1, carry=0
        vec![0.0, 1.0], // sum=0, carry=1
    ];

    println!("Half Adder Truth Table:");
    println!("  A  B | Sum Carry");
    println!("  -----|----------");
    for (input, target) in adder_inputs.iter().zip(&adder_targets) {
        println!("  {}  {} |  {}   {}",
            input[0] as u8, input[1] as u8, target[0] as u8, target[1] as u8);
    }
    println!();
    println!("Note: Sum = XOR(A,B), Carry = AND(A,B)\n");

    // Create network: 2 inputs → 4 hidden → 2 outputs
    println!("Creating network: 2 inputs → 4 hidden → 2 outputs\n");
    let mut network = FeedForwardNetwork::new(2, 4, 2);

    // Save initial state
    println!("--- Initial Network (Random Weights) ---");
    test_network(&mut network, &adder_inputs, &adder_targets, "Initial")?;
    save_checkpoint(
        &network,
        &format!("{}/half_adder_initial.json", checkpoint_dir),
        &format!("{}/half_adder_initial.svg", image_dir),
        NetworkMetadata::initial("Half Adder Network"),
    )?;

    // Train the network
    println!("\n--- Training ---");
    println!("Learning rate: 0.5");
    println!("Target error: 0.1");
    println!("Max iterations: 5000\n");

    let iterations = network.train_by_error(&adder_inputs, &adder_targets, 0.5, Some(0.1), Some(5000))?;

    println!("\nTraining completed in {} iterations", iterations);

    // Test trained network
    println!("\n--- Trained Network ---");
    test_network(&mut network, &adder_inputs, &adder_targets, "Trained")?;
    save_checkpoint(
        &network,
        &format!("{}/half_adder_trained.json", checkpoint_dir),
        &format!("{}/half_adder_trained.svg", image_dir),
        NetworkMetadata::checkpoint("Half Adder Network - Trained", iterations, None),
    )?;

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("1. Neural networks can learn multiple outputs simultaneously");
    println!("2. Half adder demonstrates Sum = XOR(A,B) and Carry = AND(A,B)");
    println!("3. Hidden layer learns shared features useful for both outputs");
    println!("4. Multi-output networks are fundamental for complex digital circuits");
    println!("5. This is a building block for full adders and arithmetic units");
    println!("\nGenerated files:");
    println!("  - checkpoints/half_adder_*.json (network state)");
    println!("  - images/half_adder_*.svg (visualizations)");

    Ok(())
}

/// Test the network on half adder inputs and display results
fn test_network(
    network: &mut FeedForwardNetwork,
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
    stage_name: &str,
) -> Result<()> {
    println!("Testing: {}", stage_name);
    println!("  Input  Target(S,C)  Output(S,C)    Error");
    println!("  -----  -----------  ------------  --------");

    let mut total_error = 0.0;

    for (input, target) in inputs.iter().zip(targets) {
        let output = network.forward(input)?;
        let error_sum = (output[0] - target[0]).abs();
        let error_carry = (output[1] - target[1]).abs();
        let error = error_sum + error_carry;
        total_error += error;

        println!(
            "  {} {}   ({:.1},{:.1})     ({:.4},{:.4})  {:.4}",
            input[0] as u8, input[1] as u8,
            target[0], target[1],
            output[0], output[1],
            error
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
    fn test_half_adder_untrained_has_high_error() {
        // Negative test: Untrained network should produce high error
        let mut network = FeedForwardNetwork::new(2, 4, 2);

        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];

        let mut total_error = 0.0;
        for (input, target) in inputs.iter().zip(&targets) {
            let output = network.forward(input).unwrap();
            total_error += (output[0] - target[0]).abs();
            total_error += (output[1] - target[1]).abs();
        }
        let mean_error = total_error / inputs.len() as f32;

        assert!(
            mean_error > 0.4,
            "Untrained network should have high error (>0.4), but got {:.4}",
            mean_error
        );
    }

    #[test]
    fn test_half_adder_truth_table() {
        // Verify our truth table is correct
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let expected = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];

        for (input, expected_output) in inputs.iter().zip(&expected) {
            let a = input[0] as u8;
            let b = input[1] as u8;
            let sum = (a ^ b) as f32;      // XOR for sum
            let carry = (a & b) as f32;    // AND for carry
            assert_eq!(
                vec![sum, carry], *expected_output,
                "Half adder of {:?} should be {:?}",
                input, expected_output
            );
        }
    }

    #[test]
    fn test_half_adder_network_trains() {
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];

        let mut network = FeedForwardNetwork::new(2, 4, 2);
        let iterations = network
            .train_by_error(&inputs, &targets, 0.5, Some(0.1), Some(5000))
            .unwrap();

        assert!(iterations > 0, "Should train for at least 1 iteration");
        assert!(iterations <= 5000, "Should complete within max iterations");

        // Verify trained network is accurate for both outputs
        for (input, target) in inputs.iter().zip(&targets) {
            let output = network.forward(input).unwrap();
            let error_sum = (output[0] - target[0]).abs();
            let error_carry = (output[1] - target[1]).abs();
            assert!(
                error_sum < 0.6 && error_carry < 0.6,
                "Outputs ({:.4}, {:.4}) should be close to targets ({}, {})",
                output[0], output[1], target[0], target[1]
            );
        }
    }
}
