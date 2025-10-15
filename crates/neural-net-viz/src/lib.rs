//! Neural Network Visualization Library
//!
//! This crate provides SVG-based visualization for neural networks, designed for
//! educational purposes and visual learning. It generates scalable vector graphics
//! showing network architecture, weights, and training progress.
//!
//! ## Architecture
//!
//! - **Data structures** (Layer, Network, errors) → `neural-net-types` crate
//! - **Visualization** (SVG generation) → this crate (`neural-net-viz`)
//! - **Algorithms** (forward/backward propagation) → `neural-net-core` crate
//!
//! This crate depends on `neural-net-types` for data structures and adds visualization
//! capabilities via extension traits.
//!
//! ## Features
//!
//! - SVG generation for network architecture diagrams
//! - Weight visualization (color and thickness based on values)
//! - Metadata overlays (epochs, accuracy, descriptions)
//! - Configurable rendering (dimensions, colors, styles)
//! - File export for checkpoints and documentation
//!
//! ## Example
//!
//! ```
//! use neural_net_types::{FeedForwardNetwork, NetworkMetadata};
//! use neural_net_viz::{NetworkVisualization, VisualizationConfig};
//!
//! let network = FeedForwardNetwork::new(2, 4, 1);
//! let config = VisualizationConfig::default();
//! let metadata = NetworkMetadata::initial("XOR Network");
//!
//! // Generate SVG
//! let svg = network.to_svg(&config).unwrap();
//!
//! // Or with metadata overlay
//! let svg_with_meta = network.to_svg_with_metadata(&metadata, &config).unwrap();
//!
//! // Verify SVG was generated
//! assert!(svg.contains("<svg"));
//! assert!(svg_with_meta.contains("XOR Network"));
//! ```

use neural_net_types::{FeedForwardNetwork, NetworkMetadata, Result};
use std::fmt::Write;
use std::fs::File;
use std::io::Write as IoWrite;
use std::path::Path;

/// Configuration for SVG rendering
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Width of the SVG canvas
    pub width: u32,
    /// Height of the SVG canvas
    pub height: u32,
    /// Radius of neuron circles
    pub neuron_radius: f32,
    /// Maximum weight line thickness
    pub max_weight_thickness: f32,
    /// Minimum weight line thickness
    pub min_weight_thickness: f32,
    /// Show weight values as text
    pub show_weight_values: bool,
    /// Font size for labels
    pub font_size: u32,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            width: 1200,
            height: 800,
            neuron_radius: 20.0,
            max_weight_thickness: 4.0,
            min_weight_thickness: 0.5,
            show_weight_values: false,
            font_size: 14,
        }
    }
}

/// Extension trait for FeedForwardNetwork to add visualization capabilities
pub trait NetworkVisualization {
    /// Generate an SVG visualization of the network
    fn to_svg(&self, config: &VisualizationConfig) -> Result<String>;

    /// Generate SVG with metadata overlay
    fn to_svg_with_metadata(
        &self,
        metadata: &NetworkMetadata,
        config: &VisualizationConfig,
    ) -> Result<String>;

    /// Save SVG to file
    fn save_svg(&self, path: impl AsRef<Path>, config: &VisualizationConfig) -> Result<()>;

    /// Save SVG with metadata to file
    fn save_svg_with_metadata(
        &self,
        path: impl AsRef<Path>,
        metadata: &NetworkMetadata,
        config: &VisualizationConfig,
    ) -> Result<()>;
}

impl NetworkVisualization for FeedForwardNetwork {
    fn to_svg(&self, config: &VisualizationConfig) -> Result<String> {
        let mut svg = String::new();

        // SVG header
        writeln!(&mut svg, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>").unwrap();
        writeln!(
            &mut svg,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
            config.width, config.height, config.width, config.height
        ).unwrap();

        // Background
        writeln!(&mut svg, "<rect width=\"100%\" height=\"100%\" fill=\"#f9f9f9\"/>").unwrap();

        // Calculate layer positions
        let layer_positions = calculate_layer_positions(self, config);

        // Draw connections first (so neurons appear on top)
        draw_connections(self, &mut svg, config, &layer_positions);

        // Draw neurons
        draw_neurons(self, &mut svg, config, &layer_positions);

        // Draw labels
        draw_labels(self, &mut svg, config, &layer_positions);

        // Close SVG
        writeln!(&mut svg, "</svg>").unwrap();

        Ok(svg)
    }

    fn to_svg_with_metadata(
        &self,
        metadata: &NetworkMetadata,
        config: &VisualizationConfig,
    ) -> Result<String> {
        let mut svg = self.to_svg(config)?;

        // Insert metadata before closing </svg>
        let insert_pos = svg.rfind("</svg>").unwrap();
        let metadata_svg = generate_metadata_overlay(metadata, config);
        svg.insert_str(insert_pos, &metadata_svg);

        Ok(svg)
    }

    fn save_svg(&self, path: impl AsRef<Path>, config: &VisualizationConfig) -> Result<()> {
        let svg = self.to_svg(config)?;
        let mut file = File::create(path)?;
        file.write_all(svg.as_bytes())?;
        Ok(())
    }

    fn save_svg_with_metadata(
        &self,
        path: impl AsRef<Path>,
        metadata: &NetworkMetadata,
        config: &VisualizationConfig,
    ) -> Result<()> {
        let svg = self.to_svg_with_metadata(metadata, config)?;
        let mut file = File::create(path)?;
        file.write_all(svg.as_bytes())?;
        Ok(())
    }
}

/// Calculate (x, y) positions for all neurons in all layers
fn calculate_layer_positions(network: &FeedForwardNetwork, config: &VisualizationConfig) -> Vec<Vec<(f32, f32)>> {
    let num_layers = network.layer_count();
    let margin_x = 100.0;
    let margin_y = 100.0;
    let usable_width = config.width as f32 - 2.0 * margin_x;
    let usable_height = config.height as f32 - 2.0 * margin_y;

    let layer_spacing = usable_width / (num_layers - 1) as f32;

    let mut positions = Vec::new();

    for layer_idx in 0..num_layers {
        let layer = network.layer(layer_idx).unwrap();
        let num_neurons = layer.num_neurons();

        let x = margin_x + layer_idx as f32 * layer_spacing;
        let neuron_spacing = if num_neurons > 1 {
            usable_height / (num_neurons - 1) as f32
        } else {
            0.0
        };

        let mut layer_positions = Vec::new();
        for neuron_idx in 0..num_neurons {
            let y = if num_neurons == 1 {
                config.height as f32 / 2.0
            } else {
                margin_y + neuron_idx as f32 * neuron_spacing
            };
            layer_positions.push((x, y));
        }
        positions.push(layer_positions);
    }

    positions
}

/// Draw connection lines between neurons
fn draw_connections(
    network: &FeedForwardNetwork,
    svg: &mut String,
    config: &VisualizationConfig,
    positions: &[Vec<(f32, f32)>],
) {
    writeln!(svg, "<g id=\"connections\">").unwrap();

    for layer_idx in 1..network.layer_count() {
        let layer = network.layer(layer_idx).unwrap();
        if let Some(weights) = layer.weights() {
            let prev_positions = &positions[layer_idx - 1];
            let curr_positions = &positions[layer_idx];

            for (from_idx, from_pos) in prev_positions.iter().enumerate() {
                for (to_idx, to_pos) in curr_positions.iter().enumerate() {
                    let weight = weights[[from_idx, to_idx]];
                    let thickness = weight_to_thickness(weight, config);
                    let color = if weight >= 0.0 { "#4CAF50" } else { "#F44336" };
                    let opacity = weight.abs().min(1.0) * 0.6 + 0.1;

                    writeln!(
                        svg,
                        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\" opacity=\"{}\"/>",
                        from_pos.0, from_pos.1, to_pos.0, to_pos.1, color, thickness, opacity
                    ).unwrap();
                }
            }
        }
    }

    writeln!(svg, "</g>").unwrap();
}

/// Draw neuron circles
fn draw_neurons(
    network: &FeedForwardNetwork,
    svg: &mut String,
    config: &VisualizationConfig,
    positions: &[Vec<(f32, f32)>],
) {
    writeln!(svg, "<g id=\"neurons\">").unwrap();

    for (layer_idx, layer_positions) in positions.iter().enumerate() {
        let fill = match layer_idx {
            0 => "#2196F3",      // Input: blue
            idx if idx == network.layer_count() - 1 => "#FF9800", // Output: orange
            _ => "#9C27B0",      // Hidden: purple
        };

        for (neuron_idx, (x, y)) in layer_positions.iter().enumerate() {
            writeln!(
                svg,
                "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"{}\" stroke=\"#333\" stroke-width=\"2\"/>",
                x, y, config.neuron_radius, fill
            ).unwrap();

            // Neuron label
            writeln!(
                svg,
                "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" dominant-baseline=\"middle\" fill=\"white\" font-size=\"{}\">#{}</text>",
                x, y, config.font_size - 2, neuron_idx
            ).unwrap();
        }
    }

    writeln!(svg, "</g>").unwrap();
}

/// Draw layer labels
fn draw_labels(
    network: &FeedForwardNetwork,
    svg: &mut String,
    config: &VisualizationConfig,
    positions: &[Vec<(f32, f32)>],
) {
    writeln!(svg, "<g id=\"labels\">").unwrap();

    for (layer_idx, layer_positions) in positions.iter().enumerate() {
        let layer = network.layer(layer_idx).unwrap();
        let x = layer_positions[0].0;
        let y = 50.0;

        let label = match layer_idx {
            0 => format!("Input Layer\n{} neurons", layer.num_neurons()),
            idx if idx == network.layer_count() - 1 => {
                format!("Output Layer\n{} neurons", layer.num_neurons())
            }
            _ => format!("Hidden Layer\n{} neurons", layer.num_neurons()),
        };

        for (i, line) in label.lines().enumerate() {
            writeln!(
                svg,
                "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-size=\"{}\" font-weight=\"bold\" fill=\"#333\">{}</text>",
                x, y + i as f32 * 20.0, config.font_size, line
            ).unwrap();
        }
    }

    writeln!(svg, "</g>").unwrap();
}

/// Generate metadata overlay
fn generate_metadata_overlay(
    metadata: &NetworkMetadata,
    config: &VisualizationConfig,
) -> String {
    let mut svg = String::new();
    writeln!(svg, "<g id=\"metadata\">").unwrap();

    // Background box
    writeln!(
        svg,
        "<rect x=\"10\" y=\"{}\" width=\"300\" height=\"120\" fill=\"white\" stroke=\"#333\" stroke-width=\"2\" rx=\"5\"/>",
        config.height - 130
    ).unwrap();

    let base_y = config.height - 115;
    let mut y_offset = 0;

    writeln!(
        svg,
        "<text x=\"20\" y=\"{}\" font-size=\"16\" font-weight=\"bold\" fill=\"#333\">{}</text>",
        base_y + y_offset,
        metadata.name
    ).unwrap();
    y_offset += 25;

    writeln!(
        svg,
        "<text x=\"20\" y=\"{}\" font-size=\"12\" fill=\"#666\">Epochs: {}</text>",
        base_y + y_offset,
        metadata.epochs
    ).unwrap();
    y_offset += 20;

    if let Some(accuracy) = metadata.accuracy {
        writeln!(
            svg,
            "<text x=\"20\" y=\"{}\" font-size=\"12\" fill=\"#666\">Accuracy: {:.2}%</text>",
            base_y + y_offset,
            accuracy
        ).unwrap();
        y_offset += 20;
    }

    writeln!(
        svg,
        "<text x=\"20\" y=\"{}\" font-size=\"10\" fill=\"#999\">{}</text>",
        base_y + y_offset,
        metadata.description
    ).unwrap();

    writeln!(svg, "</g>").unwrap();
    svg
}

/// Convert weight value to line thickness
fn weight_to_thickness(weight: f32, config: &VisualizationConfig) -> f32 {
    let abs_weight = weight.abs();
    let normalized = abs_weight.min(1.0); // Cap at 1.0
    config.min_weight_thickness
        + normalized * (config.max_weight_thickness - config.min_weight_thickness)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svg_generation() {
        let network = FeedForwardNetwork::new(2, 3, 1);
        let config = VisualizationConfig::default();

        let svg = network.to_svg(&config).unwrap();

        assert!(svg.contains("<?xml"));
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("connections"));
        assert!(svg.contains("neurons"));
    }

    #[test]
    fn test_svg_with_metadata() {
        let network = FeedForwardNetwork::new(2, 4, 1);
        let metadata = NetworkMetadata::checkpoint("XOR", 1000, Some(95.5));
        let config = VisualizationConfig::default();

        let svg = network.to_svg_with_metadata(&metadata, &config).unwrap();

        assert!(svg.contains("XOR"));
        assert!(svg.contains("1000"));
        assert!(svg.contains("95.5"));
    }

    #[test]
    fn test_save_svg() {
        use std::env;

        let network = FeedForwardNetwork::new(2, 3, 1);
        let config = VisualizationConfig::default();
        let temp_path = env::temp_dir().join("test_network.svg");

        network.save_svg(&temp_path, &config).unwrap();

        // Verify file exists and contains SVG
        let content = std::fs::read_to_string(&temp_path).unwrap();
        assert!(content.contains("<svg"));

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }
}
