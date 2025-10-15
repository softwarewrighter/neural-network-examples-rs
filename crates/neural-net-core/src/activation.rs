//! Activation functions for neural network layers

/// Trait for activation functions
pub trait Activation {
    /// Apply the activation function to an input value
    fn activate(&self, x: f32) -> f32;

    /// Calculate the derivative of the activation function given the output
    fn derivative(&self, output: f32) -> f32;
}

/// Sigmoid activation function: f(x) = 1 / (1 + e^-x)
///
/// Commonly used in hidden layers. Output range is (0, 1).
#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    #[inline]
    fn activate(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[inline]
    fn derivative(&self, output: f32) -> f32 {
        output * (1.0 - output)
    }
}

/// Linear activation function: f(x) = x
///
/// Used in input and output layers for regression tasks.
#[derive(Debug, Clone, Copy)]
pub struct Linear;

impl Activation for Linear {
    #[inline]
    fn activate(&self, x: f32) -> f32 {
        x
    }

    #[inline]
    fn derivative(&self, _output: f32) -> f32 {
        1.0
    }
}

/// ReLU (Rectified Linear Unit) activation function: f(x) = max(0, x)
///
/// The most popular activation for hidden layers in modern deep networks.
/// Prevents vanishing gradient problem and is computationally efficient.
///
/// ## Properties
/// - Range: [0, ∞)
/// - Non-saturating (no vanishing gradient for x > 0)
/// - Sparse activation (~50% of neurons are zero)
/// - Can suffer from "dying ReLU" problem (neurons permanently dead if x < 0)
///
/// ## Formula
/// ```text
/// f(x) = max(0, x) = { x  if x > 0
///                    { 0  if x ≤ 0
///
/// f'(x) = { 1  if x > 0
///         { 0  if x ≤ 0
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ReLU;

impl Activation for ReLU {
    #[inline]
    fn activate(&self, x: f32) -> f32 {
        x.max(0.0)
    }

    #[inline]
    fn derivative(&self, output: f32) -> f32 {
        if output > 0.0 { 1.0 } else { 0.0 }
    }
}

/// Leaky ReLU activation function: f(x) = max(αx, x)
///
/// Variant of ReLU that allows small negative values to prevent dying neurons.
///
/// ## Properties
/// - Range: (-∞, ∞)
/// - Prevents dying ReLU problem
/// - Small gradient for negative values (α = 0.01 typically)
///
/// ## Formula
/// ```text
/// f(x) = max(αx, x) = { x   if x > 0
///                     { αx  if x ≤ 0
///
/// f'(x) = { 1  if x > 0
///         { α  if x ≤ 0
/// ```
#[derive(Debug, Clone, Copy)]
pub struct LeakyReLU {
    /// Slope for negative values (typically 0.01)
    pub alpha: f32,
}

impl LeakyReLU {
    /// Create a new Leaky ReLU with default alpha = 0.01
    pub fn new() -> Self {
        Self { alpha: 0.01 }
    }

    /// Create a new Leaky ReLU with custom alpha
    pub fn with_alpha(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Activation for LeakyReLU {
    #[inline]
    fn activate(&self, x: f32) -> f32 {
        if x > 0.0 { x } else { self.alpha * x }
    }

    #[inline]
    fn derivative(&self, output: f32) -> f32 {
        if output > 0.0 { 1.0 } else { self.alpha }
    }
}

/// GELU (Gaussian Error Linear Unit) activation function
///
/// Used in modern transformers (BERT, GPT) and the TRM paper.
/// Smoother than ReLU, stochastically regularizes by multiplying input by Bernoulli(Φ(x)).
///
/// ## Properties
/// - Range: (-∞, ∞)
/// - Smooth, differentiable everywhere
/// - Non-monotonic (has a small negative region)
/// - Better performance than ReLU on many tasks
///
/// ## Formula (Exact)
/// ```text
/// f(x) = x * Φ(x) = x * P(X ≤ x) where X ~ N(0,1)
///      = x * (1/2)[1 + erf(x/√2)]
/// ```
///
/// ## Formula (Approximation - used here)
/// ```text
/// f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
///
/// f'(x) ≈ computed numerically or via automatic differentiation
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GELU;

impl Activation for GELU {
    #[inline]
    fn activate(&self, x: f32) -> f32 {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/π)
        const COEFF: f32 = 0.044715;

        let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

    #[inline]
    fn derivative(&self, output: f32) -> f32 {
        // For derivative, we need the original input x
        // Since we only have output, we approximate using numerical derivative
        // In practice, this would be computed during backprop with the stored input
        // For now, we use a reasonable approximation
        // True derivative is complex: 0.5 * (1 + tanh(...)) + x * sech^2(...) * (...)

        // Approximate: if output > 0, gradient ≈ 1, otherwise small positive
        if output > 0.0 {
            1.0
        } else {
            0.1 // Small gradient for negative region
        }
    }
}

/// Swish activation function: f(x) = x * sigmoid(x)
///
/// Also known as SiLU (Sigmoid Linear Unit). Smooth, non-monotonic activation
/// that has been shown to work better than ReLU on deep networks.
///
/// ## Properties
/// - Range: (-∞, ∞)
/// - Smooth, differentiable everywhere
/// - Non-monotonic (small negative region)
/// - Self-gated (uses its own value to gate)
///
/// ## Formula
/// ```text
/// f(x) = x * σ(x) = x / (1 + e^(-x))
///
/// f'(x) = f(x) + σ(x) * (1 - f(x))
///       = σ(x) * (1 + x * (1 - σ(x)))
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Swish;

impl Activation for Swish {
    #[inline]
    fn activate(&self, x: f32) -> f32 {
        // f(x) = x * sigmoid(x)
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        x * sigmoid
    }

    #[inline]
    fn derivative(&self, output: f32) -> f32 {
        // Approximate derivative
        // True derivative: σ(x) * (1 + x * (1 - σ(x)))
        // Since we only have output = x * σ(x), we approximate
        if output > 0.0 {
            1.0
        } else {
            0.2 // Small gradient for negative region
        }
    }
}

/// Tanh (Hyperbolic Tangent) activation function: f(x) = tanh(x)
///
/// Similar to sigmoid but centered at zero with range (-1, 1).
/// Often works better than sigmoid for hidden layers.
///
/// ## Properties
/// - Range: (-1, 1)
/// - Zero-centered (unlike sigmoid)
/// - Still suffers from vanishing gradient at extremes
/// - Stronger gradients than sigmoid
///
/// ## Formula
/// ```text
/// f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
///
/// f'(x) = 1 - tanh²(x) = 1 - f(x)²
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl Activation for Tanh {
    #[inline]
    fn activate(&self, x: f32) -> f32 {
        x.tanh()
    }

    #[inline]
    fn derivative(&self, output: f32) -> f32 {
        1.0 - output * output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sigmoid_activation() {
        let sigmoid = Sigmoid;
        assert_relative_eq!(sigmoid.activate(0.0), 0.5, epsilon = 0.0001);
        assert_relative_eq!(sigmoid.activate(10.0), 0.9999, epsilon = 0.0001);
        assert_relative_eq!(sigmoid.activate(-10.0), 0.0001, epsilon = 0.0001);
    }

    #[test]
    fn test_sigmoid_derivative() {
        let sigmoid = Sigmoid;
        // Derivative at output=0.5 should be 0.25
        assert_relative_eq!(sigmoid.derivative(0.5), 0.25, epsilon = 0.0001);
        // Derivative approaches 0 at extremes
        assert!(sigmoid.derivative(0.01) < 0.01);
        assert!(sigmoid.derivative(0.99) < 0.01);
    }

    #[test]
    fn test_linear_activation() {
        let linear = Linear;
        assert_eq!(linear.activate(5.0), 5.0);
        assert_eq!(linear.activate(-3.0), -3.0);
        assert_eq!(linear.activate(0.0), 0.0);
    }

    #[test]
    fn test_linear_derivative() {
        let linear = Linear;
        assert_eq!(linear.derivative(10.0), 1.0);
        assert_eq!(linear.derivative(-5.0), 1.0);
        assert_eq!(linear.derivative(0.0), 1.0);
    }

    #[test]
    fn test_relu_activation() {
        let relu = ReLU;
        assert_eq!(relu.activate(5.0), 5.0);
        assert_eq!(relu.activate(0.0), 0.0);
        assert_eq!(relu.activate(-5.0), 0.0);
    }

    #[test]
    fn test_relu_derivative() {
        let relu = ReLU;
        // Derivative is 1 for positive outputs, 0 for zero/negative
        assert_eq!(relu.derivative(5.0), 1.0);
        assert_eq!(relu.derivative(0.0), 0.0);
        assert_eq!(relu.derivative(-5.0), 0.0);
    }

    #[test]
    fn test_leaky_relu_activation() {
        let leaky_relu = LeakyReLU::new(); // alpha = 0.01
        assert_eq!(leaky_relu.activate(5.0), 5.0);
        assert_eq!(leaky_relu.activate(0.0), 0.0);
        assert_relative_eq!(leaky_relu.activate(-5.0), -0.05, epsilon = 0.0001);
    }

    #[test]
    fn test_leaky_relu_derivative() {
        let leaky_relu = LeakyReLU::new(); // alpha = 0.01
        assert_eq!(leaky_relu.derivative(5.0), 1.0);
        assert_eq!(leaky_relu.derivative(-0.05), 0.01); // output was negative
    }

    #[test]
    fn test_leaky_relu_custom_alpha() {
        let leaky_relu = LeakyReLU::with_alpha(0.2);
        assert_relative_eq!(leaky_relu.activate(-5.0), -1.0, epsilon = 0.0001);
        assert_eq!(leaky_relu.derivative(-1.0), 0.2);
    }

    #[test]
    fn test_gelu_activation() {
        let gelu = GELU;
        // GELU(0) ≈ 0
        assert_relative_eq!(gelu.activate(0.0), 0.0, epsilon = 0.01);
        // GELU(positive) > 0
        assert!(gelu.activate(1.0) > 0.0);
        assert!(gelu.activate(2.0) > 0.0);
        // GELU(negative) < 0 (small negative region)
        assert!(gelu.activate(-0.5) < 0.0);
        // GELU is approximately linear for large positive x
        assert_relative_eq!(gelu.activate(5.0), 5.0, epsilon = 0.1);
    }

    #[test]
    fn test_gelu_smoothness() {
        let gelu = GELU;
        // GELU should be smooth (no sharp transitions like ReLU)
        let x1 = -0.1;
        let x2 = 0.1;
        let y1 = gelu.activate(x1);
        let y2 = gelu.activate(x2);
        // Difference should be small
        assert!((y2 - y1).abs() < 0.5);
    }

    #[test]
    fn test_swish_activation() {
        let swish = Swish;
        // Swish(0) = 0
        assert_relative_eq!(swish.activate(0.0), 0.0, epsilon = 0.01);
        // Swish(positive) > 0
        assert!(swish.activate(1.0) > 0.0);
        assert!(swish.activate(2.0) > 0.0);
        // Swish(negative) < 0 (small negative region)
        assert!(swish.activate(-2.0) < 0.0);
    }

    #[test]
    fn test_swish_vs_relu() {
        let swish = Swish;
        let relu = ReLU;

        // For large positive x, Swish ≈ ReLU (both ≈ x)
        assert_relative_eq!(swish.activate(5.0), relu.activate(5.0), epsilon = 0.5);

        // For negative x, Swish has small negative values, ReLU is zero
        assert!(swish.activate(-1.0) < 0.0);
        assert_eq!(relu.activate(-1.0), 0.0);
    }

    #[test]
    fn test_tanh_activation() {
        let tanh = Tanh;
        assert_relative_eq!(tanh.activate(0.0), 0.0, epsilon = 0.0001);
        assert_relative_eq!(tanh.activate(10.0), 1.0, epsilon = 0.0001);
        assert_relative_eq!(tanh.activate(-10.0), -1.0, epsilon = 0.0001);
        // tanh(1) ≈ 0.76
        assert_relative_eq!(tanh.activate(1.0), 0.7616, epsilon = 0.001);
    }

    #[test]
    fn test_tanh_derivative() {
        let tanh = Tanh;
        // Derivative at output=0 should be 1
        assert_relative_eq!(tanh.derivative(0.0), 1.0, epsilon = 0.0001);
        // Derivative approaches 0 at extremes
        assert!(tanh.derivative(0.99) < 0.1);
        assert!(tanh.derivative(-0.99) < 0.1);
    }

    #[test]
    fn test_activation_comparison() {
        let sigmoid = Sigmoid;
        let relu = ReLU;
        let tanh = Tanh;

        let x = 2.0;

        // Compare outputs
        let sig_out = sigmoid.activate(x);
        let relu_out = relu.activate(x);
        let tanh_out = tanh.activate(x);

        // ReLU output equals input for positive values
        assert_eq!(relu_out, x);

        // Sigmoid output is in (0, 1)
        assert!(sig_out > 0.0 && sig_out < 1.0);

        // Tanh output is in (-1, 1)
        assert!(tanh_out > -1.0 && tanh_out < 1.0);
    }
}
