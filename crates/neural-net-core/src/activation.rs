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
}
