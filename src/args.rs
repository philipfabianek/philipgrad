use clap::{error::ErrorKind, Parser};
use std::path::PathBuf;

/// Command line arguments for specifying dataset paths.
/// All paths must point to existing CSV files.
#[derive(Parser, Debug)]
struct Args {
    /// Path to training labels CSV file
    #[arg(long, value_parser = validate_file)]
    train_labels: PathBuf,

    /// Path to training vectors CSV file
    #[arg(long, value_parser = validate_file)]
    train_vectors: PathBuf,

    /// Path to test labels CSV file
    #[arg(long, value_parser = validate_file)]
    test_labels: PathBuf,

    /// Path to test vectors CSV file
    #[arg(long, value_parser = validate_file)]
    test_vectors: PathBuf,
}

#[derive(Debug)]
pub struct DatasetPaths {
    pub train_labels: String,
    pub train_vectors: String,
    pub test_labels: String,
    pub test_vectors: String,
}

/// Validates that a path points to an existing CSV file
fn validate_file(path: &str) -> Result<PathBuf, clap::Error> {
    let path = PathBuf::from(path);
    if !path.exists() {
        return Err(clap::Error::raw(
            ErrorKind::InvalidValue,
            format!("File not found: {}", path.display()),
        ));
    }
    if !path.is_file() {
        return Err(clap::Error::raw(
            ErrorKind::InvalidValue,
            format!("Not a file: {}", path.display()),
        ));
    }
    if path.extension().and_then(|s| s.to_str()) != Some("csv") {
        return Err(clap::Error::raw(
            ErrorKind::InvalidValue,
            format!("File must be a CSV: {}", path.display()),
        ));
    }
    Ok(path)
}

/// Parses and validates command line arguments
pub fn parse_arguments() -> Result<DatasetPaths, clap::Error> {
    let args = Args::parse();

    Ok(DatasetPaths {
        train_labels: args.train_labels.to_string_lossy().to_string(),
        train_vectors: args.train_vectors.to_string_lossy().to_string(),
        test_labels: args.test_labels.to_string_lossy().to_string(),
        test_vectors: args.test_vectors.to_string_lossy().to_string(),
    })
}
