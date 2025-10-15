//! File I/O utilities for reading training data

use crate::Result;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Read a matrix of floats from a text file
///
/// The file should contain space-separated float values, one row per line.
///
/// # Arguments
///
/// * `path` - Path to the file to read
///
/// # Examples
///
/// ```no_run
/// use neural_net_core::utils::read_matrix_from_file;
///
/// let matrix = read_matrix_from_file("samples/Xapp.txt").unwrap();
/// println!("Loaded matrix with {} rows", matrix.len());
/// ```
pub fn read_matrix_from_file(path: &str) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut matrix = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let row: Vec<f32> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<f32>().ok())
            .collect();

        if !row.is_empty() {
            matrix.push(row);
        }
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_read_matrix_from_file() {
        // Create a temporary test file
        let temp_path = "/tmp/test_neural_net_matrix.txt";
        let mut file = File::create(temp_path).unwrap();
        writeln!(file, "1.0 2.0 3.0").unwrap();
        writeln!(file, "4.0 5.0 6.0").unwrap();
        writeln!(file, "7.0 8.0 9.0").unwrap();

        let matrix = read_matrix_from_file(temp_path).unwrap();

        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(matrix[1], vec![4.0, 5.0, 6.0]);
        assert_eq!(matrix[2], vec![7.0, 8.0, 9.0]);

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_read_matrix_empty_lines() {
        let temp_path = "/tmp/test_neural_net_matrix_empty.txt";
        let mut file = File::create(temp_path).unwrap();
        writeln!(file, "1.0 2.0").unwrap();
        writeln!(file).unwrap(); // Empty line
        writeln!(file, "3.0 4.0").unwrap();

        let matrix = read_matrix_from_file(temp_path).unwrap();

        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0], vec![1.0, 2.0]);
        assert_eq!(matrix[1], vec![3.0, 4.0]);

        std::fs::remove_file(temp_path).ok();
    }
}
