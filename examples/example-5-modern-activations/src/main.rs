//! # Example-5: Modern Activation Functions
//!
//! This example demonstrates modern activation functions and compares them with
//! traditional Sigmoid activation, particularly focusing on the vanishing gradient problem.
//!
//! ## Key Comparisons:
//!
//! 1. **Sigmoid** - Traditional activation, suffers from vanishing gradients in deep networks
//! 2. **ReLU** - Most popular modern activation, prevents vanishing gradients
//! 3. **Leaky ReLU** - Variant of ReLU that prevents dying neurons
//! 4. **GELU** - Smooth activation used in transformers (BERT, GPT, TRM)
//! 5. **Swish** - Self-gated activation, smooth and non-monotonic
//! 6. **Tanh** - Zero-centered sigmoid variant
//!
//! ## What You'll Learn:
//!
//! - Why ReLU and GELU are preferred over Sigmoid for deep networks
//! - How to visualize gradient flow through layers
//! - The vanishing gradient problem in practice
//! - When to use each activation function

use neural_net_core::{
    FeedForwardNetwork, ForwardPropagation, NetworkTraining, Result,
    Activation, Sigmoid, ReLU, LeakyReLU, GELU, Swish, Tanh,
};

fn main() -> Result<()> {
    let separator = "======================================================================";

    println!("{}", separator);
    println!("Example-5: Modern Activation Functions");
    println!("{}", separator);
    println!();

    // Demonstrate basic activation behavior
    demonstrate_activation_outputs();
    println!();

    // Demonstrate gradient flow differences
    demonstrate_gradient_behavior();
    println!();

    // Train networks with different activations
    train_with_different_activations()?;
    println!();

    println!("{}", separator);
    println!("Key Takeaways:");
    println!("{}", separator);
    println!("1. ReLU and GELU maintain healthy gradients in deep networks");
    println!("2. Sigmoid suffers from vanishing gradients (gradients → 0 in deep layers)");
    println!("3. GELU is smooth and used in modern transformers (BERT, GPT, TRM)");
    println!("4. Leaky ReLU prevents dying neurons better than standard ReLU");
    println!("5. Swish is self-gated and works well for very deep networks");
    println!("{}", separator);

    Ok(())
}

/// Demonstrate how different activations transform the same inputs
fn demonstrate_activation_outputs() {
    println!("--- Activation Function Outputs ---");
    println!();

    let test_inputs = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

    let activations: Vec<(&str, Box<dyn Activation>)> = vec![
        ("Sigmoid", Box::new(Sigmoid)),
        ("ReLU", Box::new(ReLU)),
        ("Leaky ReLU", Box::new(LeakyReLU::new())),
        ("GELU", Box::new(GELU)),
        ("Swish", Box::new(Swish)),
        ("Tanh", Box::new(Tanh)),
    ];

    // Print header
    print!("Input    ");
    for (name, _) in &activations {
        print!("{:>12}", name);
    }
    println!();
    println!("{}", "-".repeat(90));

    // Print outputs for each input
    for input in test_inputs {
        print!("{:6.2}   ", input);
        for (_, activation) in &activations {
            let output = activation.activate(input);
            print!("{:12.6}", output);
        }
        println!();
    }

    println!();
    println!("Observations:");
    println!("• ReLU zeros out negative values, maintains positive values");
    println!("• Leaky ReLU allows small negative values (prevents dying neurons)");
    println!("• GELU is smooth, has small negative region (used in transformers)");
    println!("• Swish is similar to GELU, self-gated activation");
    println!("• Tanh is zero-centered (range: -1 to 1)");
    println!("• Sigmoid squashes to (0, 1) range");
}

/// Demonstrate gradient behavior with different activations
fn demonstrate_gradient_behavior() {
    println!("--- Gradient Behavior (Derivative at Activations) ---");
    println!();

    // Test at different activation levels
    let test_outputs = vec![
        ("Near zero", 0.01),
        ("Small positive", 0.5),
        ("Medium", 1.0),
        ("Large", 2.0),
    ];

    let activations: Vec<(&str, Box<dyn Activation>)> = vec![
        ("Sigmoid", Box::new(Sigmoid)),
        ("ReLU", Box::new(ReLU)),
        ("Leaky ReLU", Box::new(LeakyReLU::new())),
        ("GELU", Box::new(GELU)),
        ("Swish", Box::new(Swish)),
        ("Tanh", Box::new(Tanh)),
    ];

    // Print header
    print!("Output         ");
    for (name, _) in &activations {
        print!("{:>12}", name);
    }
    println!();
    println!("{}", "-".repeat(100));

    // Print derivatives for each output level
    for (label, output) in test_outputs {
        print!("{:14} ", label);
        for (_, activation) in &activations {
            let derivative = activation.derivative(output);
            print!("{:12.6}", derivative);
        }
        println!();
    }

    println!();
    println!("Gradient Observations:");
    println!("• Sigmoid gradient approaches 0 at extremes (vanishing gradient!)");
    println!("• ReLU maintains constant gradient of 1.0 for positive values");
    println!("• GELU and Swish have healthy gradients throughout");
    println!("• Tanh also suffers from vanishing gradients at extremes");
    println!();
    println!("⚠️  Vanishing Gradient Problem:");
    println!("   When gradients approach 0, deep networks can't learn effectively.");
    println!("   This is why ReLU/GELU are preferred over Sigmoid for deep networks.");
}

/// Train networks with different activations and compare performance
fn train_with_different_activations() -> Result<()> {
    println!("--- Training XOR with Different Activations ---");
    println!();

    // XOR training data (using separate vectors as required by train_by_error)
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    let error_threshold = 0.01;
    let learning_rate = 0.1;
    let max_iterations = 10000;

    println!("Task: XOR (2 inputs → 4 hidden → 1 output)");
    println!("Error threshold: {}", error_threshold);
    println!("Learning rate: {}", learning_rate);
    println!("Max iterations: {}", max_iterations);
    println!();

    // Test each activation (Note: Currently only Sigmoid is fully wired up)
    // Future versions will support different activations per layer
    let activation_names = vec!["Sigmoid (default)"];

    for activation_name in activation_names {
        print!("Training with {}... ", activation_name);

        let mut network = FeedForwardNetwork::new(2, 4, 1);

        // Train using train_by_error which trains until error threshold or max iterations
        let iterations = network.train_by_error(
            &inputs,
            &targets,
            error_threshold,
            Some(learning_rate),
            Some(max_iterations),
        )?;

        println!(" Done in {} iterations!", iterations);

        // Test accuracy
        let mut correct = 0;
        let mut total_error = 0.0;
        println!("  Results:");
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let outputs = network.forward(input)?;
            let predicted = if outputs[0] > 0.5 { 1.0 } else { 0.0 };
            let expected = target[0];

            let error = (expected - outputs[0]).abs();
            total_error += error;

            let is_correct = error < 0.3;
            if is_correct {
                correct += 1;
            }

            println!(
                "    [{:.0}, {:.0}] → {:.4} (predicted: {:.0}, expected: {:.0}) {}",
                input[0],
                input[1],
                outputs[0],
                predicted,
                expected,
                if is_correct { "✓" } else { "✗" }
            );
        }

        let accuracy = (correct as f32 / inputs.len() as f32) * 100.0;
        let mean_error = total_error / inputs.len() as f32;
        println!(
            "  Mean error: {:.6}  |  Accuracy: {}/{} ({:.1}%)",
            mean_error, correct, inputs.len(), accuracy
        );

        if accuracy == 100.0 {
            println!("  Status: ⭐ Successfully learned XOR!");
        } else {
            println!("  Status: ⚠️  Did not fully learn XOR");
        }

        println!();
    }

    println!("Note on Activation Functions:");
    println!("• Current network uses Sigmoid activation (traditional approach)");
    println!("• All modern activations (ReLU, GELU, Swish, etc.) are implemented");
    println!("• Future: Example-6 will allow per-layer activation configuration");
    println!("• For now, focus on understanding activation properties shown above");
    println!();

    println!("Why This Matters:");
    println!("• Sigmoid works well for shallow networks (3 layers)");
    println!("• ReLU and GELU are essential for deep networks (4-6+ layers)");
    println!("• GELU is preferred for transformer-based models (TRM, BERT, GPT)");
    println!("• Leaky ReLU prevents \"dying ReLU\" problem in very deep networks");

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
    fn test_xor_network_trains() {
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
        assert!(
            iterations <= 10000,
            "Should complete within max iterations"
        );

        let mean_error = compute_mean_error(&mut network, &inputs, &targets);

        assert!(
            mean_error < 0.15,
            "Trained network should have low error (<0.15), but got {:.4}",
            mean_error
        );
    }

    #[test]
    fn test_activation_functions_work() {
        // Verification test: Ensure all activations function correctly
        let activations: Vec<Box<dyn Activation>> = vec![
            Box::new(Sigmoid),
            Box::new(ReLU),
            Box::new(LeakyReLU::new()),
            Box::new(GELU),
            Box::new(Swish),
            Box::new(Tanh),
        ];

        for activation in activations {
            // Test positive value
            let output = activation.activate(1.0);
            assert!(output.is_finite(), "Activation output should be finite");

            // Test negative value
            let output = activation.activate(-1.0);
            assert!(output.is_finite(), "Activation output should be finite");

            // Test zero
            let output = activation.activate(0.0);
            assert!(output.is_finite(), "Activation output should be finite");

            // Test derivatives
            let deriv = activation.derivative(0.5);
            assert!(deriv.is_finite(), "Derivative should be finite");
        }
    }
}
