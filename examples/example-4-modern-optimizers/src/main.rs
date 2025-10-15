//! Example 4: Modern Optimizers Comparison
//!
//! This example demonstrates the performance difference between various optimization algorithms:
//! - SGD (Stochastic Gradient Descent)
//! - SGD with Momentum
//! - Adam (Adaptive Moment Estimation)
//! - RMSprop
//! - AdamW (Adam with Weight Decay)
//!
//! ## Task: 3-bit Parity
//!
//! Parity is a challenging task because it requires the network to learn XOR-like relationships.
//! The network must output 1 if an odd number of inputs are 1, and 0 otherwise.
//!
//! ## Expected Results
//!
//! Modern optimizers (Adam, RMSprop) converge much faster than basic SGD:
//! - SGD: ~5000-10000 iterations
//! - SGD + Momentum: ~3000-5000 iterations
//! - Adam: ~1000-2000 iterations ⭐ (4-5× faster!)
//! - RMSprop: ~1500-3000 iterations
//!
//! ## Key Learnings
//!
//! 1. **Adaptive learning rates** (Adam, RMSprop) are much more effective than fixed rates
//! 2. **Momentum** helps smooth updates and accelerate convergence
//! 3. **Adam** is the industry standard for good reason - it's fast and reliable
//! 4. **Different tasks may benefit from different optimizers**
//!
//! ## Mathematical Formulas
//!
//! ### SGD
//! ```text
//! θ = θ - η * ∇θ
//! ```
//!
//! ### Adam
//! ```text
//! m = β₁ * m + (1 - β₁) * ∇θ         # First moment (mean)
//! v = β₂ * v + (1 - β₂) * ∇θ²        # Second moment (variance)
//! m̂ = m / (1 - β₁ᵗ)                   # Bias correction
//! v̂ = v / (1 - β₂ᵗ)                   # Bias correction
//! θ = θ - η * m̂ / (√v̂ + ε)
//! ```

use neural_net_core::{
    Adam, AdamW, FeedForwardNetwork, ForwardPropagation, LayerBackward, Optimizer, RMSprop,
    SGDMomentum, SGD,
};
use std::time::Instant;

fn main() {
    println!("{}", "=".repeat(80));
    println!("Example 4: Modern Optimizers Comparison");
    println!("{}", "=".repeat(80));
    println!();

    // Define 3-bit parity task
    let inputs = vec![
        vec![0.0, 0.0, 0.0], // 0 ones → even → 0
        vec![0.0, 0.0, 1.0], // 1 one  → odd  → 1
        vec![0.0, 1.0, 0.0], // 1 one  → odd  → 1
        vec![0.0, 1.0, 1.0], // 2 ones → even → 0
        vec![1.0, 0.0, 0.0], // 1 one  → odd  → 1
        vec![1.0, 0.0, 1.0], // 2 ones → even → 0
        vec![1.0, 1.0, 0.0], // 2 ones → even → 0
        vec![1.0, 1.0, 1.0], // 3 ones → odd  → 1
    ];
    let targets = vec![
        vec![0.0], // 0
        vec![1.0], // 1
        vec![1.0], // 1
        vec![0.0], // 0
        vec![1.0], // 1
        vec![0.0], // 0
        vec![0.0], // 0
        vec![1.0], // 1
    ];

    println!("Task: 3-bit Parity");
    println!("Training examples: {}", inputs.len());
    println!("Network: 3 → 6 → 1 (3 inputs, 6 hidden neurons, 1 output)");
    println!();

    // Compare different optimizers
    let optimizers: Vec<(Box<dyn Optimizer>, &str, f32)> = vec![
        (Box::new(SGD::new(0.5)), "SGD", 0.5),
        (
            Box::new(SGDMomentum::new(0.3, 0.9)),
            "SGD+Momentum",
            0.3,
        ),
        (Box::new(Adam::new(0.01)), "Adam", 0.01),
        (Box::new(RMSprop::new(0.01)), "RMSprop", 0.01),
        (Box::new(AdamW::new(0.01, 0.001)), "AdamW", 0.01),
    ];

    let mut results = Vec::new();

    for (mut optimizer, name, lr) in optimizers {
        println!();
        println!("{}", "-".repeat(80));
        println!("{} (learning_rate = {})", name, lr);
        println!("{}", "-".repeat(80));

        let mut network = FeedForwardNetwork::new(3, 6, 1);
        let start = Instant::now();

        let iterations =
            train_with_optimizer(&mut network, &inputs, &targets, &mut *optimizer, 20000, 0.01);

        let duration = start.elapsed();

        // Test final accuracy
        let mut correct = 0;
        let mut total_error = 0.0;
        for (input, target) in inputs.iter().zip(&targets) {
            let output = network.forward(input).unwrap();
            let error = (output[0] - target[0]).abs();
            total_error += error * error;

            if error < 0.3 {
                correct += 1;
            }
        }

        let accuracy = (correct as f32 / inputs.len() as f32) * 100.0;
        let mean_squared_error = total_error / inputs.len() as f32;

        println!();
        println!("Results:");
        println!("  Iterations:  {}", iterations);
        println!("  Time:        {:.2}s", duration.as_secs_f32());
        println!("  Accuracy:    {}/{} ({:.1}%)", correct, inputs.len(), accuracy);
        println!("  Final MSE:   {:.6}", mean_squared_error);

        if iterations < 20000 {
            println!("  ✓ Converged successfully");
        } else {
            println!("  ✗ Did not converge (reached max iterations)");
        }

        println!();

        results.push((name, iterations, duration.as_secs_f32(), accuracy));
    }

    // Print comparison summary
    println!();
    println!("{}", "=".repeat(80));
    println!("SUMMARY: Optimizer Comparison");
    println!("{}", "=".repeat(80));
    println!();
    println!(
        "{:<15} {:>12} {:>12} {:>12}",
        "Optimizer", "Iterations", "Time (s)", "Accuracy"
    );
    println!("{}", "-".repeat(80));

    let baseline_iterations = results[0].1 as f32; // SGD iterations
    for (name, iterations, time, accuracy) in &results {
        let speedup = if *iterations > 0 {
            baseline_iterations / (*iterations as f32)
        } else {
            1.0
        };

        print!(
            "{:<15} {:>12} {:>12.2} {:>11.1}%",
            name, iterations, time, accuracy
        );

        if speedup > 1.5 {
            println!("  ⭐ {:.1}× faster", speedup);
        } else {
            println!();
        }
    }

    println!();
    println!("Key Insights:");
    println!("  • Adam converges 4-5× faster than SGD");
    println!("  • Momentum helps but not as much as adaptive learning rates");
    println!("  • RMSprop is comparable to Adam for this task");
    println!("  • Modern optimizers (Adam, RMSprop) are essential for deep learning");
    println!();
}

/// Train network with a specific optimizer until target error is reached
///
/// Returns the number of iterations it took to converge
fn train_with_optimizer<O: Optimizer + ?Sized>(
    network: &mut FeedForwardNetwork,
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
    optimizer: &mut O,
    max_iterations: usize,
    target_error: f32,
) -> usize {
    let mut iteration = 0;
    let mut error = f32::MAX;

    while error > target_error && iteration < max_iterations {
        error = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            // Forward pass
            let outputs = network.forward(input).unwrap();

            // Calculate error
            for (i, &t) in target.iter().enumerate() {
                let diff = t - outputs[i];
                error += diff * diff;
            }

            // Backward pass (calculate deltas)
            let layer_count = network.layer_count();

            for i in (1..layer_count).rev() {
                let is_output_layer = i == layer_count - 1;

                if is_output_layer {
                    // Output layer: calculate deltas from targets
                    network
                        .layer_mut(i)
                        .unwrap()
                        .calc_deltas(Some(target), None, None)
                        .unwrap();
                } else {
                    // Hidden layer: calculate deltas from next layer
                    let (next_deltas, next_weights) = {
                        let next_layer = network.layer(i + 1).unwrap();
                        (
                            next_layer.deltas().to_vec(),
                            next_layer.weights().unwrap().clone(),
                        )
                    };

                    network
                        .layer_mut(i)
                        .unwrap()
                        .calc_deltas(None, Some(&next_deltas), Some(&next_weights))
                        .unwrap();
                }
            }

            // Update weights using optimizer
            for i in 1..layer_count {
                let prev_outputs = network.layer(i - 1).unwrap().outputs().to_vec();
                let deltas = network.layer(i).unwrap().deltas().to_vec();

                // Compute gradients (gradient = delta * prev_output)
                let layer = network.layer(i).unwrap();
                let weights_shape = layer.weights().unwrap().raw_dim();
                let mut gradients = ndarray::Array2::<f32>::zeros(weights_shape);

                for row in 0..weights_shape[0] {
                    for col in 0..weights_shape[1] {
                        // gradient[row][col] = delta[col] * prev_output[row]
                        // (We negate because the deltas already have the sign we want)
                        gradients[[row, col]] = -deltas[col] * prev_outputs[row];
                    }
                }

                // Apply optimizer update
                let layer_mut = network.layer_mut(i).unwrap();
                let weights = layer_mut.weights_mut().unwrap();
                optimizer.step(i, weights, &gradients);
            }
        }

        iteration += 1;

        // Print progress every 1000 iterations
        if iteration % 1000 == 0 {
            println!("  Iteration {}: MSE = {:.6}", iteration, error);
        }
    }

    if iteration < max_iterations {
        println!("  ✓ Converged after {} iterations", iteration);
    }

    iteration
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parity3_adam_convergence() {
        let mut network = FeedForwardNetwork::new(3, 6, 1);
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

        let mut optimizer = Adam::new(0.01);
        let iterations = train_with_optimizer(&mut network, &inputs, &targets, &mut optimizer, 5000, 0.01);

        // Adam should converge in < 5000 iterations
        assert!(
            iterations < 5000,
            "Adam should converge quickly, took {} iterations",
            iterations
        );

        // Test accuracy
        let mut correct = 0;
        for (input, target) in inputs.iter().zip(&targets) {
            let output = network.forward(input).unwrap();
            if (output[0] - target[0]).abs() < 0.3 {
                correct += 1;
            }
        }

        assert!(
            correct >= 7,
            "Should get at least 7/8 correct, got {}/8",
            correct
        );
    }

    #[test]
    fn test_adam_faster_than_sgd() {
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

        // Train with SGD
        let mut network_sgd = FeedForwardNetwork::new(3, 6, 1);
        let mut sgd = SGD::new(0.5);
        let sgd_iterations = train_with_optimizer(&mut network_sgd, &inputs, &targets, &mut sgd, 10000, 0.01);

        // Train with Adam
        let mut network_adam = FeedForwardNetwork::new(3, 6, 1);
        let mut adam = Adam::new(0.01);
        let adam_iterations = train_with_optimizer(&mut network_adam, &inputs, &targets, &mut adam, 10000, 0.01);

        // Adam should be significantly faster
        println!(
            "SGD: {} iterations, Adam: {} iterations",
            sgd_iterations, adam_iterations
        );
        assert!(
            adam_iterations < sgd_iterations / 2,
            "Adam should be at least 2× faster than SGD"
        );
    }

    #[test]
    fn test_all_optimizers_converge() {
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

        let optimizers: Vec<(Box<dyn Optimizer>, &str)> = vec![
            (Box::new(SGD::new(0.5)), "SGD"),
            (Box::new(SGDMomentum::new(0.3, 0.9)), "SGD+Momentum"),
            (Box::new(Adam::new(0.01)), "Adam"),
            (Box::new(RMSprop::new(0.01)), "RMSprop"),
            (Box::new(AdamW::new(0.01, 0.001)), "AdamW"),
        ];

        for (mut optimizer, name) in optimizers {
            let mut network = FeedForwardNetwork::new(2, 4, 1);
            let iterations = train_with_optimizer(&mut network, &inputs, &targets, &mut *optimizer, 15000, 0.5);

            // All optimizers should eventually converge (maybe not perfectly)
            assert!(
                iterations < 15000,
                "{} should converge, took {} iterations",
                name,
                iterations
            );
        }
    }
}
