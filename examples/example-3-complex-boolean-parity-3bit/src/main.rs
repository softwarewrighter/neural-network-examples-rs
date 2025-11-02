//! 3-Bit Parity Example - Complex Boolean Function
//!
//! This example demonstrates learning a more complex boolean function: 3-bit parity.
//! Parity checks if there's an odd number of 1s in the input.
//!
//! ## The 3-Bit Parity Problem
//!
//! Truth table:
//! | A | B | C | Parity (odd?) |
//! |---|---|---|---------------|
//! | 0 | 0 | 0 |      0        |
//! | 0 | 0 | 1 |      1        |
//! | 0 | 1 | 0 |      1        |
//! | 0 | 1 | 1 |      0        |
//! | 1 | 0 | 0 |      1        |
//! | 1 | 0 | 1 |      0        |
//! | 1 | 1 | 0 |      0        |
//! | 1 | 1 | 1 |      1        |
//!
//! ## What This Example Teaches
//!
//! 1. **Complexity Scaling**: 3-bit parity is harder than 2-bit (XOR)
//! 2. **Hidden Layer Size**: May need more hidden neurons for complex functions
//! 3. **Training Difficulty**: More inputs = larger search space = more iterations
//! 4. **Generalization**: Can the network learn the underlying pattern?
//! 5. **Parity Chain**: 3-bit parity = XOR(XOR(A, B), C)
//!
//! ## Network Architecture
//!
//! - **Input Layer**: 3 neurons (A, B, C)
//! - **Hidden Layer**: 6 neurons (more complex function needs more capacity)
//! - **Output Layer**: 1 neuron (parity bit)
//!
//! ## Expected Training Time
//!
//! - Iterations: ~3000-5000 (more complex than XOR)
//! - Learning rate: 0.5 (standard)
//! - Target error: <0.1

use neural_net_core::{
    FeedForwardNetwork, ForwardPropagation, NetworkMetadata, NetworkTraining, Result,
};
use neural_net_viz::{NetworkVisualization, VisualizationConfig};
use std::fs;

fn main() -> Result<()> {
    println!("=== 3-Bit Parity Problem ===\n");

    // Create output directories in the example's own directory
    let example_dir = env!("CARGO_MANIFEST_DIR");
    let checkpoint_dir = format!("{}/checkpoints", example_dir);
    let image_dir = format!("{}/images", example_dir);
    fs::create_dir_all(&checkpoint_dir)?;
    fs::create_dir_all(&image_dir)?;

    // Define 3-bit parity training data
    let parity_inputs = vec![
        vec![0.0, 0.0, 0.0], // 0 ones -> even -> 0
        vec![0.0, 0.0, 1.0], // 1 one  -> odd  -> 1
        vec![0.0, 1.0, 0.0], // 1 one  -> odd  -> 1
        vec![0.0, 1.0, 1.0], // 2 ones -> even -> 0
        vec![1.0, 0.0, 0.0], // 1 one  -> odd  -> 1
        vec![1.0, 0.0, 1.0], // 2 ones -> even -> 0
        vec![1.0, 1.0, 0.0], // 2 ones -> even -> 0
        vec![1.0, 1.0, 1.0], // 3 ones -> odd  -> 1
    ];
    let parity_targets = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![0.0],
        vec![1.0],
    ];

    println!("3-Bit Parity Truth Table:");
    println!("  A  B  C | Parity");
    println!("  --------|-------");
    for (input, target) in parity_inputs.iter().zip(&parity_targets) {
        println!(
            "  {}  {}  {} |   {}",
            input[0] as u8, input[1] as u8, input[2] as u8, target[0] as u8
        );
    }
    println!();

    // Create network: 3 inputs → 6 hidden → 1 output
    println!("Creating network: 3 inputs → 6 hidden → 1 output\n");
    let mut network = FeedForwardNetwork::new(3, 6, 1);

    // Save initial state
    println!("--- Initial Network (Random Weights) ---");
    test_network(&mut network, &parity_inputs, &parity_targets, "Initial")?;
    save_checkpoint(
        &network,
        &format!("{}/parity3_initial.json", checkpoint_dir),
        &format!("{}/parity3_initial.svg", image_dir),
        NetworkMetadata::initial("3-Bit Parity Network"),
    )?;

    // Train the network
    println!("\n--- Training ---");
    println!("Learning rate: 0.5");
    println!("Target error: 0.1");
    println!("Max iterations: 10000\n");

    let iterations =
        network.train_by_error(&parity_inputs, &parity_targets, 0.5, Some(0.1), Some(10000))?;

    println!("\nTraining completed in {} iterations", iterations);

    // Test trained network
    println!("\n--- Trained Network ---");
    test_network(&mut network, &parity_inputs, &parity_targets, "Trained")?;
    save_checkpoint(
        &network,
        &format!("{}/parity3_trained.json", checkpoint_dir),
        &format!("{}/parity3_trained.svg", image_dir),
        NetworkMetadata::checkpoint("3-Bit Parity Network - Trained", iterations, None),
    )?;

    println!("\n=== Example Complete ===");
    println!("\nKey Takeaways:");
    println!("1. 3-bit parity is more complex than 2-bit (XOR)");
    println!("2. More inputs require more hidden neurons (6 vs 4 for XOR)");
    println!("3. Training takes longer (~3000-5000 iterations vs ~1500-2000 for XOR)");
    println!("4. Network learns the XOR-of-XOR pattern: parity(A,B,C) = XOR(XOR(A,B), C)");
    println!("\nGenerated files:");
    println!("  - checkpoints/parity3_*.json (network state)");
    println!("  - images/parity3_*.svg (visualizations)");

    Ok(())
}

/// Test the network on parity inputs and display results
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
    fn test_parity3_truth_table() {
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
        let expected = vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];

        for (input, &expected_output) in inputs.iter().zip(&expected) {
            let count = input.iter().filter(|&&x| x == 1.0).count();
            let parity = if count % 2 == 1 { 1.0 } else { 0.0 };
            assert_eq!(
                parity, expected_output,
                "Parity of {:?} should be {}",
                input, expected_output
            );
        }
    }

    #[test]
    fn test_parity3_untrained_has_high_error() {
        // Negative test: Untrained network should produce high error
        let mut network = FeedForwardNetwork::new(3, 6, 1);

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
            vec![1.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![0.0],
            vec![1.0],
        ];

        let mut total_error = 0.0;
        for (input, target) in inputs.iter().zip(&targets) {
            let output = network.forward(input).unwrap();
            total_error += (output[0] - target[0]).abs();
        }
        let mean_error = total_error / inputs.len() as f32;

        assert!(
            mean_error > 0.4,
            "Untrained network should have high error (>0.4), but got {:.4}",
            mean_error
        );
    }

    #[test]
    fn test_parity3_network_trains() {
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
            vec![1.0],
            vec![1.0],
            vec![0.0],
            vec![1.0],
            vec![0.0],
            vec![0.0],
            vec![1.0],
        ];

        let mut network = FeedForwardNetwork::new(3, 6, 1);
        let iterations = network
            .train_by_error(&inputs, &targets, 0.5, Some(0.1), Some(10000))
            .unwrap();

        assert!(iterations > 0, "Should train for at least 1 iteration");
        assert!(iterations <= 10000, "Should complete within max iterations");

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
