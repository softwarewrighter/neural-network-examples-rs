//! Auto-generation of animation scripts from checkpoints

use crate::script::*;
use anyhow::{Context, Result};
use neural_net_core::ForwardPropagation;
use neural_net_types::FeedForwardNetwork;
use std::path::{Path, PathBuf};

/// Configuration for script generation
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Title for the animation
    pub title: String,

    /// Description
    pub description: String,

    /// Default scene duration (seconds)
    pub scene_duration: f64,

    /// Duration for intro/outro scenes
    pub intro_duration: f64,

    /// Whether to include test results
    pub include_tests: bool,

    /// Test inputs (if including tests)
    pub test_inputs: Option<Vec<Vec<f32>>>,

    /// Test targets (if including tests)
    pub test_targets: Option<Vec<Vec<f32>>>,

    /// Truth table (if applicable)
    pub truth_table: Option<TruthTable>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            title: "Neural Network Training".to_string(),
            description: "Watch the network learn".to_string(),
            scene_duration: 2.0,
            intro_duration: 3.0,
            include_tests: true,
            test_inputs: None,
            test_targets: None,
            truth_table: None,
        }
    }
}

/// Script generator
pub struct ScriptGenerator {
    config: GeneratorConfig,
}

impl ScriptGenerator {
    /// Create a new generator with config
    pub fn new(config: GeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate script from checkpoint files
    pub fn generate_from_checkpoints(
        &self,
        checkpoint_paths: &[PathBuf],
    ) -> Result<AnimationScript> {
        anyhow::ensure!(!checkpoint_paths.is_empty(), "No checkpoints provided");

        // Load first checkpoint to get network info
        let first_network = self.load_checkpoint(&checkpoint_paths[0])?;
        let network_info = self.extract_network_info(&first_network);

        // Generate scenes
        let mut scenes = Vec::new();

        for (idx, checkpoint_path) in checkpoint_paths.iter().enumerate() {
            let is_first = idx == 0;
            let is_last = idx == checkpoint_paths.len() - 1;

            let scene_id = if is_first {
                "intro".to_string()
            } else if is_last {
                "final".to_string()
            } else {
                format!("step_{}", idx)
            };

            let duration = if is_first {
                self.config.intro_duration
            } else {
                self.config.scene_duration
            };

            // Load network and run tests if configured
            let test_results = if self.config.include_tests {
                self.run_tests(checkpoint_path)?
            } else {
                None
            };

            let network_state = NetworkState {
                checkpoint_path: checkpoint_path
                    .to_string_lossy()
                    .to_string(),
                iteration: idx * 100, // Placeholder, would be extracted from checkpoint metadata
                test_results,
                weight_data: None,
            };

            let annotations = self.generate_annotations(idx, &network_state, is_first, is_last);
            let highlights = self.generate_highlights(idx, is_first, is_last);

            let transition = if is_last {
                TransitionType::Fade
            } else {
                TransitionType::Morph
            };

            scenes.push(Scene {
                id: scene_id,
                duration,
                network_state,
                annotations,
                highlights,
                transition,
            });
        }

        Ok(AnimationScript {
            metadata: AnimationMetadata {
                title: self.config.title.clone(),
                description: self.config.description.clone(),
                author: Some("neural-net-animator".to_string()),
                version: "0.1.0".to_string(),
            },
            network_info,
            truth_table: self.config.truth_table.clone(),
            scenes,
        })
    }

    /// Load checkpoint from file
    fn load_checkpoint(&self, path: &Path) -> Result<FeedForwardNetwork> {
        let (network, _metadata) = FeedForwardNetwork::load_checkpoint(path)
            .with_context(|| format!("Failed to load checkpoint: {}", path.display()))?;
        Ok(network)
    }

    /// Extract network architecture info
    fn extract_network_info(&self, network: &FeedForwardNetwork) -> NetworkInfo {
        let input_size = network.layer(0).map(|l| l.num_neurons()).unwrap_or(0);
        let hidden_size = network.layer(1).map(|l| l.num_neurons()).unwrap_or(0);
        let output_size = network
            .layer(network.layer_count() - 1)
            .map(|l| l.num_neurons())
            .unwrap_or(0);

        NetworkInfo {
            architecture: format!("{}-{}-{}", input_size, hidden_size, output_size),
            activation: "Sigmoid".to_string(), // Default, could be extracted from metadata
            input_size,
            hidden_size,
            output_size,
        }
    }

    /// Run tests on network and generate results
    fn run_tests(&self, checkpoint_path: &Path) -> Result<Option<TestResults>> {
        let inputs = match &self.config.test_inputs {
            Some(i) => i,
            None => return Ok(None),
        };

        let targets = match &self.config.test_targets {
            Some(t) => t,
            None => return Ok(None),
        };

        let mut network = self.load_checkpoint(checkpoint_path)?;

        let mut examples = Vec::new();
        let mut correct_count = 0;
        let mut total_error = 0.0;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let output = network.forward(input)?;

            let error: f32 = output
                .iter()
                .zip(target.iter())
                .map(|(o, t)| (o - t).abs())
                .sum();

            let correct = error < 0.3; // Threshold for "correct"
            if correct {
                correct_count += 1;
            }

            total_error += error;

            examples.push(ExampleResult {
                inputs: input.clone(),
                expected: target.clone(),
                actual: output,
                correct,
                error,
            });
        }

        let accuracy = correct_count as f32 / inputs.len() as f32;
        let mean_error = total_error / inputs.len() as f32;

        Ok(Some(TestResults {
            accuracy,
            mean_error,
            examples,
        }))
    }

    /// Generate annotations for a scene
    fn generate_annotations(
        &self,
        idx: usize,
        network_state: &NetworkState,
        is_first: bool,
        is_last: bool,
    ) -> Vec<Annotation> {
        let mut annotations = Vec::new();

        if is_first {
            annotations.push(Annotation {
                annotation_type: AnnotationType::Title,
                text: "Before Training".to_string(),
                position: AnnotationPosition::Named("top".to_string()),
                style: AnnotationStyle {
                    font_size: Some("32px".to_string()),
                    color: Some("#333".to_string()),
                    weight: Some("bold".to_string()),
                },
            });

            annotations.push(Annotation {
                annotation_type: AnnotationType::Label,
                text: "Random weights".to_string(),
                position: AnnotationPosition::Named("center".to_string()),
                style: AnnotationStyle::default(),
            });
        } else if is_last {
            annotations.push(Annotation {
                annotation_type: AnnotationType::Title,
                text: "Training Complete!".to_string(),
                position: AnnotationPosition::Named("top".to_string()),
                style: AnnotationStyle {
                    font_size: Some("32px".to_string()),
                    color: Some("#27ae60".to_string()),
                    weight: Some("bold".to_string()),
                },
            });
        } else {
            annotations.push(Annotation {
                annotation_type: AnnotationType::Title,
                text: format!("Training... (Step {})", idx),
                position: AnnotationPosition::Named("top".to_string()),
                style: AnnotationStyle {
                    font_size: Some("24px".to_string()),
                    color: Some("#333".to_string()),
                    weight: None,
                },
            });
        }

        // Add metrics if available
        if let Some(ref test_results) = network_state.test_results {
            annotations.push(Annotation {
                annotation_type: AnnotationType::Metric,
                text: format!("Accuracy: {:.1}%", test_results.accuracy * 100.0),
                position: AnnotationPosition::Named("bottom-left".to_string()),
                style: AnnotationStyle::default(),
            });

            annotations.push(Annotation {
                annotation_type: AnnotationType::Metric,
                text: format!("Error: {:.4}", test_results.mean_error),
                position: AnnotationPosition::Named("bottom-left".to_string()),
                style: AnnotationStyle::default(),
            });
        }

        annotations
    }

    /// Generate highlights for a scene
    fn generate_highlights(&self, _idx: usize, is_first: bool, is_last: bool) -> Vec<Highlight> {
        let mut highlights = Vec::new();

        // Highlight weight changes in training scenes
        if !is_first && !is_last {
            highlights.push(Highlight {
                highlight_type: HighlightType::WeightChange,
                layer: None,
                threshold: Some(0.1),
            });
        }

        highlights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_config_default() {
        let config = GeneratorConfig::default();
        assert_eq!(config.scene_duration, 2.0);
        assert_eq!(config.intro_duration, 3.0);
        assert!(config.include_tests);
    }

    #[test]
    fn test_extract_network_info() {
        let generator = ScriptGenerator::new(GeneratorConfig::default());
        let network = FeedForwardNetwork::new(2, 4, 1);

        let info = generator.extract_network_info(&network);
        assert_eq!(info.architecture, "2-4-1");
        assert_eq!(info.input_size, 2);
        assert_eq!(info.hidden_size, 4);
        assert_eq!(info.output_size, 1);
    }
}
