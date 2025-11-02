//! Test to ensure all markdown files contain only ASCII characters.
//!
//! This test enforces that all .md files in the repository use only ASCII characters
//! to prevent encoding issues across different platforms and tools.

use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Check if a string contains only ASCII characters
fn is_ascii_only(content: &str) -> bool {
    content.is_ascii()
}

/// Find all non-ASCII characters in a string with their positions
fn find_non_ascii_chars(content: &str) -> Vec<(usize, char, String)> {
    let mut results = Vec::new();
    let mut byte_pos = 0;

    for (char_idx, c) in content.chars().enumerate() {
        if !c.is_ascii() {
            // Calculate line number by counting newlines up to this point
            let prefix = &content[..byte_pos];
            let line_num = prefix.matches('\n').count() + 1;

            // Calculate column by finding the last newline before this position
            let line_start = prefix.rfind('\n').map(|p| p + 1).unwrap_or(0);
            let col = prefix[line_start..].chars().count() + 1;

            results.push((
                char_idx,
                c,
                format!(
                    "line {}, col {}: '{}' (U+{:04X})",
                    line_num, col, c, c as u32
                ),
            ));
        }
        byte_pos += c.len_utf8();
    }

    results
}

/// Get a helpful suggestion for common Unicode characters
fn get_ascii_suggestion(c: char) -> String {
    match c {
        '\u{2018}' | '\u{2019}' => "Use ' (single quote) instead".to_string(),
        '\u{201C}' | '\u{201D}' => "Use \" (double quote) instead".to_string(),
        '\u{2013}' => "Use - (hyphen) instead".to_string(),
        '\u{2014}' => "Use -- (double hyphen) or - instead".to_string(),
        '\u{2192}' => "Use -> instead".to_string(),
        '\u{2190}' => "Use <- instead".to_string(),
        '\u{2026}' => "Use ... (three periods) instead".to_string(),
        '\u{00A0}' => "Use regular space instead of non-breaking space".to_string(),
        '\u{2022}' => "Use * or - for bullet points instead".to_string(),
        '\u{00D7}' => "Use x (letter x) for multiplication instead".to_string(),
        _ => "Replace with ASCII equivalent".to_string(),
    }
}

#[test]
fn test_markdown_files_are_ascii_only() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut failures = Vec::new();

    // Find all .md files in the repository
    for entry in WalkDir::new(repo_root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "md"))
    {
        let path = entry.path();

        // Skip files in target/ and .git/ directories
        if path.to_str().unwrap_or("").contains("/target/")
            || path.to_str().unwrap_or("").contains("/.git/")
        {
            continue;
        }

        let content = fs::read_to_string(path).unwrap_or_else(|e| {
            panic!("Failed to read {}: {}", path.display(), e);
        });

        if !is_ascii_only(&content) {
            let non_ascii_chars = find_non_ascii_chars(&content);
            let relative_path = path.strip_prefix(repo_root).unwrap_or(path);

            failures.push(format!(
                "\n❌ {}\n   Found {} non-ASCII character(s):",
                relative_path.display(),
                non_ascii_chars.len()
            ));

            for (_, c, location) in non_ascii_chars.iter().take(10) {
                failures.push(format!(
                    "   - {}  →  {}",
                    location,
                    get_ascii_suggestion(*c)
                ));
            }

            if non_ascii_chars.len() > 10 {
                failures.push(format!("   ... and {} more", non_ascii_chars.len() - 10));
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "\n\n⚠️  MARKDOWN ASCII CHECK FAILED\n\
             \n\
             The following markdown files contain non-ASCII characters:\
             {}\n\
             \n\
             ℹ️  All markdown files MUST use only ASCII characters to prevent encoding issues.\n\
             \n\
             Common fixes:\n\
             - Replace smart quotes (\u{201C}\u{201D}) with regular quotes (\"\")\n\
             - Replace em dashes (\u{2014}) with hyphens (-)\n\
             - Replace arrows (\u{2192}) with -> or <-\n\
             - Replace ellipses (\u{2026}) with ...\n\
             \n\
             See documentation/process.md for complete requirements.\n",
            failures.join("\n")
        );
    }
}
