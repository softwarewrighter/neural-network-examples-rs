//! Animation script data structures
//!
//! Defines the format for animation scripts that control how neural network
//! training is visualized over time.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete animation script
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationScript {
    /// Metadata about the animation
    pub metadata: AnimationMetadata,

    /// Network architecture information
    pub network_info: NetworkInfo,

    /// Optional truth table to display
    pub truth_table: Option<TruthTable>,

    /// Ordered sequence of scenes
    pub scenes: Vec<Scene>,
}

impl AnimationScript {
    /// Get total duration of the animation in seconds
    pub fn total_duration(&self) -> f64 {
        self.scenes.iter().map(|s| s.duration).sum()
    }

    /// Find scene at specific time point
    pub fn scene_at_time(&self, time: f64) -> Option<(usize, &Scene, f64)> {
        let mut cumulative_time = 0.0;

        for (idx, scene) in self.scenes.iter().enumerate() {
            if time >= cumulative_time && time < cumulative_time + scene.duration {
                let scene_time = time - cumulative_time;
                return Some((idx, scene, scene_time));
            }
            cumulative_time += scene.duration;
        }

        None
    }

    /// Get cumulative time at start of scene
    pub fn scene_start_time(&self, scene_idx: usize) -> f64 {
        self.scenes.iter()
            .take(scene_idx)
            .map(|s| s.duration)
            .sum()
    }
}

/// Animation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationMetadata {
    /// Title of the animation
    pub title: String,

    /// Description
    pub description: String,

    /// Author (optional)
    pub author: Option<String>,

    /// Version of the script format
    pub version: String,
}

impl Default for AnimationMetadata {
    fn default() -> Self {
        Self {
            title: "Untitled Animation".to_string(),
            description: String::new(),
            author: None,
            version: "0.1.0".to_string(),
        }
    }
}

/// Network architecture information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    /// Architecture description (e.g., "2-4-1")
    pub architecture: String,

    /// Activation function used
    pub activation: String,

    /// Number of input neurons
    pub input_size: usize,

    /// Number of hidden neurons
    pub hidden_size: usize,

    /// Number of output neurons
    pub output_size: usize,
}

/// Truth table for displaying expected behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthTable {
    /// Input column names
    pub input_labels: Vec<String>,

    /// Output column names
    pub output_labels: Vec<String>,

    /// Table rows
    pub rows: Vec<TruthTableRow>,
}

/// Single row in truth table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthTableRow {
    /// Input values
    pub inputs: Vec<f32>,

    /// Expected output values
    pub expected: Vec<f32>,

    /// Actual output values (filled during animation)
    pub actual: Option<Vec<f32>>,
}

/// Individual scene in the animation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    /// Unique identifier
    pub id: String,

    /// Duration in seconds
    pub duration: f64,

    /// Network state to display
    pub network_state: NetworkState,

    /// Annotations to show
    pub annotations: Vec<Annotation>,

    /// Highlights
    pub highlights: Vec<Highlight>,

    /// Transition to next scene
    pub transition: TransitionType,
}

/// Network state at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkState {
    /// Path to checkpoint file
    pub checkpoint_path: String,

    /// Iteration number
    pub iteration: usize,

    /// Test results (if available)
    pub test_results: Option<TestResults>,

    /// Optional: Specific weights to show (layer_idx -> weights JSON)
    pub weight_data: Option<HashMap<usize, String>>,
}

/// Test results at a checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    /// Overall accuracy (0.0 to 1.0)
    pub accuracy: f32,

    /// Mean error
    pub mean_error: f32,

    /// Per-example results
    pub examples: Vec<ExampleResult>,
}

/// Result for a single test example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleResult {
    /// Input values
    pub inputs: Vec<f32>,

    /// Expected output
    pub expected: Vec<f32>,

    /// Actual output
    pub actual: Vec<f32>,

    /// Whether this example is correct
    pub correct: bool,

    /// Error magnitude
    pub error: f32,
}

/// Annotation to display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Annotation {
    /// Type of annotation
    pub annotation_type: AnnotationType,

    /// Text content
    pub text: String,

    /// Position on screen
    pub position: AnnotationPosition,

    /// Style
    pub style: AnnotationStyle,
}

/// Type of annotation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnnotationType {
    Title,
    Label,
    Metric,
    Explanation,
}

/// Position of annotation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnnotationPosition {
    /// Fixed position name
    Named(String),  // "top", "bottom-left", etc.

    /// Exact coordinates
    Coords { x: f32, y: f32 },
}

/// Annotation style
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnnotationStyle {
    /// Font size
    pub font_size: Option<String>,

    /// Color
    pub color: Option<String>,

    /// Font weight
    pub weight: Option<String>,
}

/// Visual highlight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Highlight {
    /// Type of highlight
    pub highlight_type: HighlightType,

    /// Target layer (if applicable)
    pub layer: Option<usize>,

    /// Threshold for change detection (if applicable)
    pub threshold: Option<f32>,
}

/// Type of highlight
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HighlightType {
    /// Highlight weights that changed significantly
    WeightChange,

    /// Highlight specific neurons
    Neurons { indices: Vec<usize> },

    /// Highlight input/output path
    DataFlow { from: Vec<usize>, to: Vec<usize> },
}

/// Transition between scenes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TransitionType {
    /// No transition, instant cut
    Cut,

    /// Fade out/in
    Fade,

    /// Morph weights smoothly
    Morph,

    /// Slide transition
    Slide,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_script_serialization() {
        let script = AnimationScript {
            metadata: AnimationMetadata {
                title: "Test Animation".to_string(),
                description: "Test".to_string(),
                author: Some("Test".to_string()),
                version: "0.1.0".to_string(),
            },
            network_info: NetworkInfo {
                architecture: "2-4-1".to_string(),
                activation: "Sigmoid".to_string(),
                input_size: 2,
                hidden_size: 4,
                output_size: 1,
            },
            truth_table: None,
            scenes: vec![
                Scene {
                    id: "test".to_string(),
                    duration: 5.0,
                    network_state: NetworkState {
                        checkpoint_path: "test.json".to_string(),
                        iteration: 0,
                        test_results: None,
                        weight_data: None,
                    },
                    annotations: vec![],
                    highlights: vec![],
                    transition: TransitionType::Fade,
                },
            ],
        };

        let json = serde_json::to_string_pretty(&script).unwrap();
        let parsed: AnimationScript = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.metadata.title, "Test Animation");
        assert_eq!(parsed.scenes.len(), 1);
        assert_eq!(parsed.total_duration(), 5.0);
    }

    #[test]
    fn test_scene_at_time() {
        let script = AnimationScript {
            metadata: AnimationMetadata::default(),
            network_info: NetworkInfo {
                architecture: "2-4-1".to_string(),
                activation: "Sigmoid".to_string(),
                input_size: 2,
                hidden_size: 4,
                output_size: 1,
            },
            truth_table: None,
            scenes: vec![
                Scene {
                    id: "scene1".to_string(),
                    duration: 3.0,
                    network_state: NetworkState {
                        checkpoint_path: "1.json".to_string(),
                        iteration: 0,
                        test_results: None,
                        weight_data: None,
                    },
                    annotations: vec![],
                    highlights: vec![],
                    transition: TransitionType::Fade,
                },
                Scene {
                    id: "scene2".to_string(),
                    duration: 2.0,
                    network_state: NetworkState {
                        checkpoint_path: "2.json".to_string(),
                        iteration: 100,
                        test_results: None,
                        weight_data: None,
                    },
                    annotations: vec![],
                    highlights: vec![],
                    transition: TransitionType::Morph,
                },
            ],
        };

        // Test time in first scene
        let (idx, scene, time) = script.scene_at_time(1.5).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(scene.id, "scene1");
        assert!((time - 1.5).abs() < 0.001);

        // Test time in second scene
        let (idx, scene, time) = script.scene_at_time(4.0).unwrap();
        assert_eq!(idx, 1);
        assert_eq!(scene.id, "scene2");
        assert!((time - 1.0).abs() < 0.001);

        // Test time beyond animation
        assert!(script.scene_at_time(10.0).is_none());
    }
}
