//! Full Adder Example - Multi-Output with Carry Chain
//!
//! This example demonstrates learning a full adder circuit. A full adder adds
//! two binary digits plus a carry-in, producing a sum and carry-out.
//!
//! ## The Full Adder Problem
//!
//! Truth table:
//! | A | B | Cin | Sum | Cout |
//! |---|---|-----|-----|------|
//! | 0 | 0 |  0  |  0  |  0   |
//! | 0 | 0 |  1  |  1  |  0   |
//! | 0 | 1 |  0  |  1  |  0   |
//! | 0 | 1 |  1  |  0  |  1   |
//! | 1 | 0 |  0  |  1  |  0   |
//! | 1 | 0 |  1  |  0  |  1   |
//! | 1 | 1 |  0  |  0  |  1   |
//! | 1 | 1 |  1  |  1  |  1   |
//!
//! ## What This Example Teaches
//!
//! 1. **Carry Chains**: Full adders can be chained to create multi-bit adders
//! 2. **Digital Arithmetic**: Foundation of CPU arithmetic logic units (ALUs)
//! 3. **Complex Multi-Output**: 3 inputs → 2 outputs with intricate relationships
//! 4. **Building Blocks**: Full adder = 2 half adders + OR gate
//! 5. **Neural Network Capacity**: Can learn complex boolean circuits
//!
//! ## Network Architecture
//!
//! - **Input Layer**: 3 neurons (A, B, Carry-in)
//! - **Hidden Layer**: 6 neurons (more complex than half adder)
//! - **Output Layer**: 2 neurons (sum, carry-out)
//!
//! ## Expected Training Time
//!
//! - Iterations: ~3000-5000 (complex multi-output function)
//! - Learning rate: 0.5 (standard)
//! - Target error: <0.1

use neural_net_core::{FeedForwardNetwork, ForwardPropagation, NetworkTraining, NetworkMetadata, Result};
use neural_net_viz::{NetworkVisualization, VisualizationConfig};
use std::fs;

fn main() -> Result<()> {
    println!("=== Full Adder Problem ===\n");

    // Create output directories in the example's own directory
    let example_dir = env!("CARGO_MANIFEST_DIR");
    let checkpoint_dir = format!("{}/checkpoints", example_dir);
    let image_dir = format!("{}/images", example_dir);
    fs::create_dir_all(&checkpoint_dir)?;
    fs::create_dir_all(&image_dir)?;

    // Define full adder training data
    let adder_inputs = vec![
        vec![0.0, 0.0, 0.0], // 0 + 0 + 0
        vec![0.0, 0.0, 1.0], // 0 + 0 + 1
        vec![0.0, 1.0, 0.0], // 0 + 1 + 0
        vec![0.0, 1.0, 1.0], // 0 + 1 + 1
        vec![1.0, 0.0, 0.0], // 1 + 0 + 0
        vec![1.0, 0.0, 1.0], // 1 + 0 + 1
        vec![1.0, 1.0, 0.0], // 1 + 1 + 0
        vec![1.0, 1.0, 1.0], // 1 + 1 + 1
    ];
    // Each target has two outputs: [sum, carry_out]
    let adder_targets = vec![
        vec![0.0, 0.0], // sum=0, cout=0
        vec![1.0, 0.0], // sum=1, cout=0
        vec![1.0, 0.0], // sum=1, cout=0
        vec![0.0, 1.0], // sum=0, cout=1
        vec![1.0, 0.0], // sum=1, cout=0
        vec![0.0, 1.0], // sum=0, cout=1
        vec![0.0, 1.0], // sum=0, cout=1
        vec![1.0, 1.0], // sum=1, cout=1
    ];

    println!("Full Adder Truth Table:");
    println!("  A  B  Cin | Sum Cout");
    println!("  ----------|-----------");
    for (input, target) in adder_inputs.iter().zip(&adder_targets) {
        println!("  {}  {}   {} |  {}   {}",
            input[0] as u8, input[1] as u8, input[2] as u8,
            target[0] as u8, target[1] as u8);
    }
    println!();
    println!("Note: Sum = XOR(A,B,Cin), Cout = MAJ(A,B,Cin)\n");

    // Create network: 3 inputs → 6 hidden → 2 outputs
    println!("Creating network: 3 inputs → 6 hidden → 2 outputs\n");
    let mut network = FeedForwardNetwork::new(3, 6, 2);

    // Save initial state
    println!("--- Initial Network (Random Weights) ---");
    test_network(&mut network, &adder_inputs, &adder_targets, "Initial")?;
    save_checkpoint(
        &network,
        &format!("{}/full_adder_initial.json", checkpoint_dir),
        &format!("{}/full_adder_initial.svg", image_dir),
        NetworkMetadata::initial("Full Adder Network"),
    )?;

    // Train the network
    println!("\n--- Training ---");
    println!("Learning rate: 0.5");
    println!("Target error: 0.1");
    println!("Max iterations: 10000\n");

    let iterations = network.train_by_error(&adder_inputs, &adder_targets, 0.5, Some(0.1), Some(10000))?;

    println!("\nTraining completed in {} iterations", iterations);

    // Test trained network
    println!("\n--- Trained Network ---");
    test_network(&mut network, &adder_inputs, &adder_targets, "Trained")?;
    save_checkpoint(
        &network,
        &format!("{}/full_adder_trained.json", checkpoint_dir),
        &format!("{}/full_adder_trained.svg", image_dir),
        NetworkMetadata::checkpoint("Full Adder Network - Trained", iterations, None),
    )?;

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("1. Full adder combines two binary digits plus carry-in");
    println!("2. Output: Sum = 3-bit parity(A,B,Cin), Carry = majority(A,B,Cin)");
    println!("3. Full adders chain together to create multi-bit arithmetic");
    println!("4. This is the foundation of CPU arithmetic units (ALUs)");
    println!("5. Neural networks can learn complex digital logic circuits");
    println!("\nReal-world application:");
    println!("  - 8-bit adder: Chain 8 full adders");
    println!("  - 32-bit adder: Chain 32 full adders");
    println!("  - Each full adder's Cout feeds next adder's Cin");
    println!("\nGenerated files:");
    println!("  - checkpoints/full_adder_*.json (network state)");
    println!("  - images/full_adder_*.svg (visualizations)");

    Ok(())
}

/// Test the network on full adder inputs and display results
fn test_network(
    network: &mut FeedForwardNetwork,
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
    stage_name: &str,
) -> Result<()> {
    println!("Testing: {}", stage_name);
    println!("  Input    Target(S,C)  Output(S,C)    Error");
    println!("  -------  -----------  ------------  --------");

    let mut total_error = 0.0;

    for (input, target) in inputs.iter().zip(targets) {
        let output = network.forward(input)?;
        let error_sum = (output[0] - target[0]).abs();
        let error_carry = (output[1] - target[1]).abs();
        let error = error_sum + error_carry;
        total_error += error;

        println!(
            "  {} {} {}   ({:.1},{:.1})     ({:.4},{:.4})  {:.4}",
            input[0] as u8, input[1] as u8, input[2] as u8,
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
    fn test_full_adder_untrained_has_high_error() {
        // Negative test: Untrained network should produce high error
        let mut network = FeedForwardNetwork::new(3, 6, 2);

        let inputs = [vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0]];
        let targets = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
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
    fn test_full_adder_truth_table() {
        // Verify our truth table is correct
        let inputs = [vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0]];
        let expected = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        for (input, expected_output) in inputs.iter().zip(&expected) {
            let a = input[0] as u8;
            let b = input[1] as u8;
            let cin = input[2] as u8;

            // Sum is 3-bit parity (XOR of all inputs)
            let sum = ((a ^ b) ^ cin) as f32;

            // Carry-out is majority function
            let ones_count = (a + b + cin) as usize;
            let cout = if ones_count >= 2 { 1.0 } else { 0.0 };

            assert_eq!(
                vec![sum, cout], *expected_output,
                "Full adder of {:?} should be {:?}",
                input, expected_output
            );
        }
    }

    #[test]
    fn test_full_adder_network_trains() {
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
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let mut network = FeedForwardNetwork::new(3, 6, 2);
        let iterations = network
            .train_by_error(&inputs, &targets, 0.5, Some(0.1), Some(10000))
            .unwrap();

        assert!(iterations > 0, "Should train for at least 1 iteration");
        assert!(iterations <= 10000, "Should complete within max iterations");

        // Verify trained network is accurate for both outputs
        for (input, target) in inputs.iter().zip(&targets) {
            let output = network.forward(input).unwrap();
            let error_sum = (output[0] - target[0]).abs();
            let error_carry = (output[1] - target[1]).abs();
            assert!(
                error_sum < 0.5 && error_carry < 0.5,
                "Outputs ({:.4}, {:.4}) should be close to targets ({}, {})",
                output[0], output[1], target[0], target[1]
            );
        }
    }
}
