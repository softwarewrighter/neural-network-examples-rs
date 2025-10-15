//! Neural Network Animation Tool
//!
//! This crate provides tools for creating interactive animations of neural network
//! training processes. It includes:
//!
//! - Animation script format (JSON-based)
//! - Timeline management with DVR-like controls
//! - Web server for serving interactive visualizations
//! - Auto-generation of scripts from checkpoint files
//!
//! # Example
//!
//! ```no_run
//! use neural_net_animator::{GeneratorConfig, ScriptGenerator};
//! use std::path::PathBuf;
//!
//! let config = GeneratorConfig {
//!     title: "XOR Training".to_string(),
//!     description: "Watch the network learn XOR".to_string(),
//!     ..Default::default()
//! };
//!
//! let generator = ScriptGenerator::new(config);
//! let checkpoints = vec![
//!     PathBuf::from("checkpoints/xor_initial.json"),
//!     PathBuf::from("checkpoints/xor_trained.json"),
//! ];
//!
//! let script = generator.generate_from_checkpoints(&checkpoints).unwrap();
//! ```

pub mod generator;
pub mod script;
pub mod server;
pub mod timeline;

pub use generator::{GeneratorConfig, ScriptGenerator};
pub use script::{
    AnimationScript, AnimationMetadata, Annotation, AnnotationPosition, AnnotationStyle,
    AnnotationType, ExampleResult, Highlight, HighlightType, NetworkInfo, NetworkState, Scene,
    TestResults, TransitionType, TruthTable, TruthTableRow,
};
pub use timeline::{PlaybackSpeed, PlaybackState, Timeline};

/// Result type for this crate
pub type Result<T> = anyhow::Result<T>;

#[cfg(test)]
mod tests {
    #[test]
    fn test_lib_compiles() {
        // Smoke test to ensure library compiles
    }
}
