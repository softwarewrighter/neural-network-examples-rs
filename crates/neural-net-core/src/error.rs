//! Error types for neural network operations

use thiserror::Error;

/// Errors that can occur during neural network operations
#[derive(Debug, Error)]
pub enum NeuralNetError {
    /// Invalid layer configuration
    #[error("Invalid layer configuration: {0}")]
    InvalidConfig(String),

    /// Input dimension mismatch
    #[error("Input dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension received
        actual: usize,
    },

    /// IO error occurred
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Training failed
    #[error("Training failed: {0}")]
    TrainingError(String),
}

/// Result type for neural network operations
pub type Result<T> = std::result::Result<T, NeuralNetError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = NeuralNetError::DimensionMismatch {
            expected: 3,
            actual: 5,
        };
        assert!(err.to_string().contains("expected 3"));
        assert!(err.to_string().contains("got 5"));
    }

    #[test]
    fn test_invalid_config() {
        let err = NeuralNetError::InvalidConfig("test error".to_string());
        assert!(err.to_string().contains("test error"));
    }
}
