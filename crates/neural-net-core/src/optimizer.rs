//! Optimizers for neural network training
//!
//! This module provides various optimization algorithms for updating neural network weights.
//!
//! ## Optimizers
//!
//! - **SGD**: Stochastic Gradient Descent (basic)
//! - **SGD with Momentum**: Accumulates velocity to smooth updates
//! - **Adam**: Adaptive Moment Estimation (industry standard)
//! - **RMSprop**: Root Mean Square Propagation
//! - **AdamW**: Adam with decoupled weight decay
//!
//! ## Mathematical Formulas
//!
//! ### SGD
//! ```text
//! θ = θ - η * ∇θ
//! ```
//!
//! ### SGD with Momentum
//! ```text
//! v = β * v + ∇θ
//! θ = θ - η * v
//! ```
//!
//! ### Adam
//! ```text
//! m = β₁ * m + (1 - β₁) * ∇θ         # First moment (mean)
//! v = β₂ * v + (1 - β₂) * ∇θ²        # Second moment (variance)
//! m̂ = m / (1 - β₁ᵗ)                   # Bias-corrected first moment
//! v̂ = v / (1 - β₂ᵗ)                   # Bias-corrected second moment
//! θ = θ - η * m̂ / (√v̂ + ε)
//! ```
//!
//! ### RMSprop
//! ```text
//! v = β * v + (1 - β) * ∇θ²
//! θ = θ - η * ∇θ / (√v + ε)
//! ```

use ndarray::Array2;
use std::collections::HashMap;

/// Trait for optimization algorithms
pub trait Optimizer: Send {
    /// Update weights using gradients
    ///
    /// # Arguments
    ///
    /// * `param_id` - Unique identifier for this parameter (e.g., layer index)
    /// * `weights` - Mutable reference to weight matrix
    /// * `gradients` - Gradient matrix (same shape as weights)
    fn step(&mut self, param_id: usize, weights: &mut Array2<f32>, gradients: &Array2<f32>);

    /// Reset optimizer state (clears momentum, velocity, etc.)
    fn reset(&mut self);

    /// Get optimizer name for logging
    fn name(&self) -> &str;
}

/// Stochastic Gradient Descent (SGD)
///
/// The simplest optimizer: θ = θ - η * ∇θ
pub struct SGD {
    learning_rate: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, _param_id: usize, weights: &mut Array2<f32>, gradients: &Array2<f32>) {
        // θ = θ - η * ∇θ
        for i in 0..weights.shape()[0] {
            for j in 0..weights.shape()[1] {
                weights[[i, j]] -= self.learning_rate * gradients[[i, j]];
            }
        }
    }

    fn reset(&mut self) {
        // SGD has no state to reset
    }

    fn name(&self) -> &str {
        "SGD"
    }
}

/// SGD with Momentum
///
/// Accumulates a velocity vector to smooth updates and accelerate convergence:
/// ```text
/// v = β * v + ∇θ
/// θ = θ - η * v
/// ```
pub struct SGDMomentum {
    learning_rate: f32,
    momentum: f32,
    /// Velocity vectors for each parameter (indexed by param_id)
    velocity: HashMap<usize, Array2<f32>>,
}

impl SGDMomentum {
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: HashMap::new(),
        }
    }
}

impl Optimizer for SGDMomentum {
    fn step(&mut self, param_id: usize, weights: &mut Array2<f32>, gradients: &Array2<f32>) {
        // Initialize velocity if first time seeing this parameter
        let velocity = self
            .velocity
            .entry(param_id)
            .or_insert_with(|| Array2::zeros(weights.raw_dim()));

        // v = β * v + ∇θ
        for i in 0..weights.shape()[0] {
            for j in 0..weights.shape()[1] {
                velocity[[i, j]] = self.momentum * velocity[[i, j]] + gradients[[i, j]];
                // θ = θ - η * v
                weights[[i, j]] -= self.learning_rate * velocity[[i, j]];
            }
        }
    }

    fn reset(&mut self) {
        self.velocity.clear();
    }

    fn name(&self) -> &str {
        "SGD+Momentum"
    }
}

/// Adam (Adaptive Moment Estimation)
///
/// Industry-standard optimizer that combines momentum and adaptive learning rates.
/// Maintains first moment (mean) and second moment (variance) estimates.
///
/// ## Algorithm
/// ```text
/// m = β₁ * m + (1 - β₁) * ∇θ         # First moment (mean)
/// v = β₂ * v + (1 - β₂) * ∇θ²        # Second moment (variance)
/// m̂ = m / (1 - β₁ᵗ)                   # Bias correction
/// v̂ = v / (1 - β₂ᵗ)                   # Bias correction
/// θ = θ - η * m̂ / (√v̂ + ε)
/// ```
///
/// ## Default Hyperparameters
/// - β₁ = 0.9 (momentum for first moment)
/// - β₂ = 0.999 (momentum for second moment)
/// - ε = 1e-8 (small constant for numerical stability)
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    /// First moment estimates (mean) for each parameter
    m: HashMap<usize, Array2<f32>>,
    /// Second moment estimates (variance) for each parameter
    v: HashMap<usize, Array2<f32>>,
    /// Timestep counter (for bias correction)
    t: usize,
}

impl Adam {
    /// Create Adam optimizer with default hyperparameters
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate (typically 0.001)
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Create Adam optimizer with custom hyperparameters
    pub fn with_params(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, param_id: usize, weights: &mut Array2<f32>, gradients: &Array2<f32>) {
        // Increment timestep
        self.t += 1;

        // Initialize moments if first time seeing this parameter
        let m = self
            .m
            .entry(param_id)
            .or_insert_with(|| Array2::zeros(weights.raw_dim()));
        let v = self
            .v
            .entry(param_id)
            .or_insert_with(|| Array2::zeros(weights.raw_dim()));

        // Update biased first and second moment estimates
        for i in 0..weights.shape()[0] {
            for j in 0..weights.shape()[1] {
                let grad = gradients[[i, j]];

                // m = β₁ * m + (1 - β₁) * ∇θ
                m[[i, j]] = self.beta1 * m[[i, j]] + (1.0 - self.beta1) * grad;

                // v = β₂ * v + (1 - β₂) * ∇θ²
                v[[i, j]] = self.beta2 * v[[i, j]] + (1.0 - self.beta2) * grad * grad;

                // Bias-corrected first moment estimate
                let m_hat = m[[i, j]] / (1.0 - self.beta1.powi(self.t as i32));

                // Bias-corrected second moment estimate
                let v_hat = v[[i, j]] / (1.0 - self.beta2.powi(self.t as i32));

                // Update weights: θ = θ - η * m̂ / (√v̂ + ε)
                weights[[i, j]] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
        }
    }

    fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }

    fn name(&self) -> &str {
        "Adam"
    }
}

/// RMSprop (Root Mean Square Propagation)
///
/// Adaptive learning rate optimizer that divides the learning rate by
/// a running average of the magnitudes of recent gradients.
///
/// ## Algorithm
/// ```text
/// v = β * v + (1 - β) * ∇θ²
/// θ = θ - η * ∇θ / (√v + ε)
/// ```
pub struct RMSprop {
    learning_rate: f32,
    beta: f32,
    epsilon: f32,
    /// Running average of squared gradients for each parameter
    v: HashMap<usize, Array2<f32>>,
}

impl RMSprop {
    /// Create RMSprop optimizer with default hyperparameters
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate (typically 0.001)
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta: 0.9,
            epsilon: 1e-8,
            v: HashMap::new(),
        }
    }

    /// Create RMSprop optimizer with custom hyperparameters
    pub fn with_params(learning_rate: f32, beta: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta,
            epsilon,
            v: HashMap::new(),
        }
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, param_id: usize, weights: &mut Array2<f32>, gradients: &Array2<f32>) {
        // Initialize running average if first time seeing this parameter
        let v = self
            .v
            .entry(param_id)
            .or_insert_with(|| Array2::zeros(weights.raw_dim()));

        // Update running average and weights
        for i in 0..weights.shape()[0] {
            for j in 0..weights.shape()[1] {
                let grad = gradients[[i, j]];

                // v = β * v + (1 - β) * ∇θ²
                v[[i, j]] = self.beta * v[[i, j]] + (1.0 - self.beta) * grad * grad;

                // θ = θ - η * ∇θ / (√v + ε)
                weights[[i, j]] -= self.learning_rate * grad / (v[[i, j]].sqrt() + self.epsilon);
            }
        }
    }

    fn reset(&mut self) {
        self.v.clear();
    }

    fn name(&self) -> &str {
        "RMSprop"
    }
}

/// AdamW (Adam with decoupled Weight decay)
///
/// Variant of Adam that decouples weight decay from the gradient-based update.
/// This improves regularization compared to L2 regularization in Adam.
///
/// ## Algorithm
/// ```text
/// m = β₁ * m + (1 - β₁) * ∇θ
/// v = β₂ * v + (1 - β₂) * ∇θ²
/// m̂ = m / (1 - β₁ᵗ)
/// v̂ = v / (1 - β₂ᵗ)
/// θ = θ - η * (m̂ / (√v̂ + ε) + λ * θ)    # Weight decay applied separately
/// ```
pub struct AdamW {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    /// First moment estimates (mean) for each parameter
    m: HashMap<usize, Array2<f32>>,
    /// Second moment estimates (variance) for each parameter
    v: HashMap<usize, Array2<f32>>,
    /// Timestep counter (for bias correction)
    t: usize,
}

impl AdamW {
    /// Create AdamW optimizer with default hyperparameters
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate (typically 0.001)
    /// * `weight_decay` - Weight decay coefficient (typically 0.01)
    pub fn new(learning_rate: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Create AdamW optimizer with custom hyperparameters
    pub fn with_params(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, param_id: usize, weights: &mut Array2<f32>, gradients: &Array2<f32>) {
        // Increment timestep
        self.t += 1;

        // Initialize moments if first time seeing this parameter
        let m = self
            .m
            .entry(param_id)
            .or_insert_with(|| Array2::zeros(weights.raw_dim()));
        let v = self
            .v
            .entry(param_id)
            .or_insert_with(|| Array2::zeros(weights.raw_dim()));

        // Update biased first and second moment estimates
        for i in 0..weights.shape()[0] {
            for j in 0..weights.shape()[1] {
                let grad = gradients[[i, j]];

                // m = β₁ * m + (1 - β₁) * ∇θ
                m[[i, j]] = self.beta1 * m[[i, j]] + (1.0 - self.beta1) * grad;

                // v = β₂ * v + (1 - β₂) * ∇θ²
                v[[i, j]] = self.beta2 * v[[i, j]] + (1.0 - self.beta2) * grad * grad;

                // Bias-corrected first moment estimate
                let m_hat = m[[i, j]] / (1.0 - self.beta1.powi(self.t as i32));

                // Bias-corrected second moment estimate
                let v_hat = v[[i, j]] / (1.0 - self.beta2.powi(self.t as i32));

                // AdamW: Apply weight decay separately from gradient update
                // θ = θ - η * (m̂ / (√v̂ + ε) + λ * θ)
                let update = m_hat / (v_hat.sqrt() + self.epsilon) + self.weight_decay * weights[[i, j]];
                weights[[i, j]] -= self.learning_rate * update;
            }
        }
    }

    fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }

    fn name(&self) -> &str {
        "AdamW"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sgd_step() {
        let mut optimizer = SGD::new(0.1);
        let mut weights = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let gradients = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        optimizer.step(0, &mut weights, &gradients);

        // θ = θ - η * ∇θ
        // weights[0][0] = 1.0 - 0.1 * 0.1 = 0.99
        assert_relative_eq!(weights[[0, 0]], 0.99, epsilon = 1e-6);
        assert_relative_eq!(weights[[0, 1]], 1.98, epsilon = 1e-6);
        assert_relative_eq!(weights[[1, 0]], 2.97, epsilon = 1e-6);
        assert_relative_eq!(weights[[1, 1]], 3.96, epsilon = 1e-6);
    }

    #[test]
    fn test_sgd_momentum_accumulation() {
        let mut optimizer = SGDMomentum::new(0.1, 0.9);
        let mut weights = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let gradients = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

        // First step: v = 0.9*0 + 1.0 = 1.0, θ = 1.0 - 0.1*1.0 = 0.9
        optimizer.step(0, &mut weights, &gradients);
        assert_relative_eq!(weights[[0, 0]], 0.9, epsilon = 1e-6);

        // Second step: v = 0.9*1.0 + 1.0 = 1.9, θ = 0.9 - 0.1*1.9 = 0.71
        optimizer.step(0, &mut weights, &gradients);
        assert_relative_eq!(weights[[0, 0]], 0.71, epsilon = 1e-6);
    }

    #[test]
    fn test_adam_bias_correction() {
        let mut optimizer = Adam::new(0.1);
        let mut weights = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let gradients = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

        // First step should apply bias correction
        optimizer.step(0, &mut weights, &gradients);

        // m = 0.9*0 + 0.1*1.0 = 0.1
        // v = 0.999*0 + 0.001*1.0 = 0.001
        // m_hat = 0.1 / (1 - 0.9^1) = 0.1 / 0.1 = 1.0
        // v_hat = 0.001 / (1 - 0.999^1) = 0.001 / 0.001 = 1.0
        // θ = 1.0 - 0.1 * 1.0 / (sqrt(1.0) + 1e-8) ≈ 0.9
        assert_relative_eq!(weights[[0, 0]], 0.9, epsilon = 1e-3);
    }

    #[test]
    fn test_rmsprop_adaptive_lr() {
        let mut optimizer = RMSprop::new(0.1);
        let mut weights = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

        // Large gradient
        let large_grad = Array2::from_shape_vec((1, 1), vec![10.0]).unwrap();
        optimizer.step(0, &mut weights, &large_grad);
        let delta_large = 1.0 - weights[[0, 0]];

        // Reset and try small gradient
        weights = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        optimizer.reset();
        let small_grad = Array2::from_shape_vec((1, 1), vec![0.1]).unwrap();
        optimizer.step(0, &mut weights, &small_grad);
        let delta_small = 1.0 - weights[[0, 0]];

        // RMSprop should take larger steps for smaller gradients (adaptive learning rate)
        // After normalization, small gradient gets boosted more
        assert!(delta_small > delta_large * 0.5); // Roughly similar magnitude after adaptation
    }

    #[test]
    fn test_adamw_weight_decay() {
        let mut optimizer = AdamW::new(0.1, 0.01);
        let mut weights = Array2::from_shape_vec((1, 1), vec![10.0]).unwrap();
        let gradients = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap();

        // With zero gradients, only weight decay should apply
        let initial_weight = weights[[0, 0]];
        optimizer.step(0, &mut weights, &gradients);

        // Weight should decrease due to weight decay
        assert!(weights[[0, 0]] < initial_weight);
    }

    #[test]
    fn test_optimizer_reset() {
        let mut optimizer = Adam::new(0.1);
        let mut weights = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let gradients = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

        // Take a step
        optimizer.step(0, &mut weights, &gradients);
        assert_eq!(optimizer.t, 1);
        assert!(optimizer.m.contains_key(&0));

        // Reset should clear state
        optimizer.reset();
        assert_eq!(optimizer.t, 0);
        assert!(optimizer.m.is_empty());
        assert!(optimizer.v.is_empty());
    }
}
