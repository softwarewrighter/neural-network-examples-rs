//! Network metadata and checkpoint management
//!
//! Provides data structures for saving/loading network state with metadata.

use crate::{FeedForwardNetwork, Result};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Metadata for a network checkpoint
///
/// Stores information about a network's state at a particular point in time,
/// useful for tracking training progress and documenting experiments.
///
/// # Examples
///
/// ```
/// use neural_net_types::NetworkMetadata;
///
/// // Create metadata for initial state
/// let meta = NetworkMetadata::initial("XOR Network");
/// assert_eq!(meta.epochs, 0);
///
/// // Create metadata for training checkpoint
/// let meta = NetworkMetadata::checkpoint("XOR Network", 1000, Some(95.5));
/// assert_eq!(meta.epochs, 1000);
/// assert_eq!(meta.accuracy, Some(95.5));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetadata {
    /// Human-readable name for this checkpoint
    pub name: String,
    /// Description of this checkpoint's purpose
    pub description: String,
    /// Timestamp when checkpoint was created (ISO 8601 format)
    pub timestamp: String,
    /// Number of training epochs completed
    pub epochs: usize,
    /// Current training accuracy (0.0 - 100.0), if available
    pub accuracy: Option<f32>,
    /// Additional custom metadata
    #[serde(default)]
    pub custom: std::collections::HashMap<String, String>,
}

impl NetworkMetadata {
    /// Create new metadata with current timestamp
    ///
    /// # Arguments
    ///
    /// * `name` - Human-readable checkpoint name
    /// * `description` - Description of checkpoint purpose
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            epochs: 0,
            accuracy: None,
            custom: std::collections::HashMap::new(),
        }
    }

    /// Create metadata for initial random weights
    ///
    /// # Examples
    ///
    /// ```
    /// use neural_net_types::NetworkMetadata;
    ///
    /// let meta = NetworkMetadata::initial("My Network");
    /// assert_eq!(meta.name, "My Network");
    /// assert_eq!(meta.epochs, 0);
    /// assert!(meta.accuracy.is_none());
    /// ```
    pub fn initial(network_name: impl Into<String>) -> Self {
        Self::new(network_name, "Initial network with random weights")
    }

    /// Create metadata for a training checkpoint
    ///
    /// # Arguments
    ///
    /// * `network_name` - Base name for the network
    /// * `epochs` - Number of epochs completed
    /// * `accuracy` - Current accuracy (optional)
    pub fn checkpoint(
        network_name: impl Into<String>,
        epochs: usize,
        accuracy: Option<f32>,
    ) -> Self {
        let mut meta = Self::new(
            format!("{} - Epoch {}", network_name.into(), epochs),
            format!("Training checkpoint after {} epochs", epochs),
        );
        meta.epochs = epochs;
        meta.accuracy = accuracy;
        meta
    }

    /// Create metadata for final trained network
    ///
    /// # Arguments
    ///
    /// * `network_name` - Base name for the network
    /// * `epochs` - Total number of epochs trained
    /// * `accuracy` - Final accuracy achieved
    pub fn final_trained(
        network_name: impl Into<String>,
        epochs: usize,
        accuracy: f32,
    ) -> Self {
        let mut meta = Self::new(
            format!("{} - Trained", network_name.into()),
            format!("Fully trained network after {} epochs", epochs),
        );
        meta.epochs = epochs;
        meta.accuracy = Some(accuracy);
        meta
    }
}

/// A complete network checkpoint with metadata
///
/// Combines a network state with descriptive metadata for storage and retrieval.
///
/// # Examples
///
/// ```
/// use neural_net_types::{FeedForwardNetwork, NetworkMetadata, NetworkCheckpoint};
/// use std::env;
///
/// let network = FeedForwardNetwork::new(2, 4, 1);
/// let metadata = NetworkMetadata::initial("Test Network");
/// let checkpoint = NetworkCheckpoint::new(network, metadata);
///
/// // Save to file
/// let temp_path = env::temp_dir().join("test_checkpoint.json");
/// checkpoint.save_to_file(&temp_path).unwrap();
///
/// // Load from file
/// let loaded = NetworkCheckpoint::load_from_file(&temp_path).unwrap();
/// assert_eq!(loaded.network.layer_count(), 3);
///
/// // Cleanup
/// std::fs::remove_file(temp_path).ok();
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkCheckpoint {
    /// Checkpoint metadata
    pub metadata: NetworkMetadata,
    /// The network state
    pub network: FeedForwardNetwork,
}

impl NetworkCheckpoint {
    /// Create a new checkpoint
    ///
    /// # Arguments
    ///
    /// * `network` - The network to checkpoint
    /// * `metadata` - Metadata describing this checkpoint
    pub fn new(network: FeedForwardNetwork, metadata: NetworkMetadata) -> Self {
        Self { metadata, network }
    }

    /// Save checkpoint to JSON file with pretty formatting
    ///
    /// # Arguments
    ///
    /// * `path` - File path to save to
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be created or JSON serialization fails
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Load checkpoint from JSON file
    ///
    /// # Arguments
    ///
    /// * `path` - File path to load from
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be opened or JSON deserialization fails
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let checkpoint = serde_json::from_reader(reader)?;
        Ok(checkpoint)
    }
}

impl FeedForwardNetwork {
    /// Save network to JSON file with metadata
    ///
    /// Convenience method that creates a checkpoint and saves it.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to save to
    /// * `metadata` - Metadata describing this checkpoint
    pub fn save_checkpoint(
        &self,
        path: impl AsRef<Path>,
        metadata: NetworkMetadata,
    ) -> Result<()> {
        let checkpoint = NetworkCheckpoint::new(self.clone(), metadata);
        checkpoint.save_to_file(path)
    }

    /// Load network from JSON checkpoint file
    ///
    /// Returns both the network and its associated metadata.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to load from
    ///
    /// # Returns
    ///
    /// Tuple of (network, metadata)
    pub fn load_checkpoint(path: impl AsRef<Path>) -> Result<(Self, NetworkMetadata)> {
        let checkpoint = NetworkCheckpoint::load_from_file(path)?;
        Ok((checkpoint.network, checkpoint.metadata))
    }

    /// Save network to JSON (without metadata wrapper)
    ///
    /// Saves just the network structure, without checkpoint metadata.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to save to
    pub fn save_to_json(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Load network from JSON (without metadata wrapper)
    ///
    /// Loads just the network structure, without checkpoint metadata.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to load from
    pub fn load_from_json(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let network = serde_json::from_reader(reader)?;
        Ok(network)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_creation() {
        let meta = NetworkMetadata::new("Test Network", "Testing metadata");
        assert_eq!(meta.name, "Test Network");
        assert_eq!(meta.description, "Testing metadata");
        assert_eq!(meta.epochs, 0);
        assert!(meta.accuracy.is_none());
        assert!(!meta.timestamp.is_empty());
    }

    #[test]
    fn test_metadata_initial() {
        let meta = NetworkMetadata::initial("XOR Network");
        assert_eq!(meta.name, "XOR Network");
        assert!(meta.description.contains("Initial"));
        assert_eq!(meta.epochs, 0);
    }

    #[test]
    fn test_metadata_checkpoint() {
        let meta = NetworkMetadata::checkpoint("XOR", 1000, Some(95.5));
        assert_eq!(meta.name, "XOR - Epoch 1000");
        assert_eq!(meta.epochs, 1000);
        assert_eq!(meta.accuracy, Some(95.5));
    }

    #[test]
    fn test_metadata_final_trained() {
        let meta = NetworkMetadata::final_trained("XOR", 5000, 98.5);
        assert!(meta.name.contains("Trained"));
        assert_eq!(meta.epochs, 5000);
        assert_eq!(meta.accuracy, Some(98.5));
    }

    #[test]
    fn test_network_serialization() {
        let network = FeedForwardNetwork::new(2, 3, 1);

        // Serialize to JSON string
        let json = serde_json::to_string_pretty(&network).unwrap();
        assert!(json.contains("layers"));

        // Deserialize back
        let loaded: FeedForwardNetwork = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.layer_count(), 3);
    }

    #[test]
    fn test_checkpoint_creation() {
        let network = FeedForwardNetwork::new(2, 4, 1);
        let metadata = NetworkMetadata::initial("Test");
        let checkpoint = NetworkCheckpoint::new(network, metadata.clone());

        assert_eq!(checkpoint.network.layer_count(), 3);
        assert_eq!(checkpoint.metadata.name, metadata.name);
    }

    #[test]
    fn test_checkpoint_save_load() {
        use std::env;

        let network = FeedForwardNetwork::new(2, 4, 1);
        let metadata = NetworkMetadata::initial("Test Network");

        let temp_path = env::temp_dir().join("test_checkpoint_types.json");

        // Save
        network.save_checkpoint(&temp_path, metadata.clone()).unwrap();

        // Load
        let (loaded_network, loaded_metadata) =
            FeedForwardNetwork::load_checkpoint(&temp_path).unwrap();

        assert_eq!(loaded_network.layer_count(), 3);
        assert_eq!(loaded_metadata.name, metadata.name);
        assert_eq!(loaded_metadata.description, metadata.description);

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_save_load_json() {
        use std::env;

        let network = FeedForwardNetwork::new(3, 5, 2);
        let temp_path = env::temp_dir().join("test_network_types.json");

        // Save
        network.save_to_json(&temp_path).unwrap();

        // Load
        let loaded = FeedForwardNetwork::load_from_json(&temp_path).unwrap();
        assert_eq!(loaded.layer_count(), network.layer_count());

        // Cleanup
        std::fs::remove_file(temp_path).ok();
    }
}
