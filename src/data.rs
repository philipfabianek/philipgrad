use crate::{parse_sample, NormalizationParams};
use rand::prelude::SliceRandom;
use rand::thread_rng;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Loads and normalizes a batch of samples from iterators
///
/// # Arguments
/// * `vectors` - Iterator over feature vectors
/// * `labels` - Iterator over labels
/// * `minibatch_size` - Desired batch size
/// * `norm_params` - Parameters for feature normalization
///
/// # Returns
/// Option containing (normalized inputs, targets) if data available
pub fn load_batch<'a>(
    vectors: &mut impl Iterator<Item = &'a String>,
    labels: &mut impl Iterator<Item = &'a String>,
    vector_size: usize,
    minibatch_size: usize,
    norm_params: &NormalizationParams,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let mut inputs = Vec::with_capacity(minibatch_size * vector_size);
    let mut targets = Vec::with_capacity(minibatch_size);

    let mean = norm_params.mean;
    let std_dev = norm_params.std_dev;

    let mut got_any = false;
    for _ in 0..minibatch_size {
        match (vectors.next(), labels.next()) {
            (Some(vector_str), Some(label_str)) => {
                got_any = true;
                let (values, target) = parse_sample(vector_str, label_str);
                // Normalize
                for v in values {
                    inputs.push((v - mean) / std_dev);
                }
                targets.push(target);
            }
            _ => break,
        }
    }

    if got_any {
        Some((inputs, targets))
    } else {
        None
    }
}

/// Loads data from a file into a vector of strings
pub fn load_data_file(file_path: &String) -> Vec<String> {
    let file = File::open(file_path).unwrap();
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().map(|l| l.unwrap()).collect();
    lines
}

/// Prepares data for iteration, optionally shuffling
///
/// # Arguments
/// * `labels` - Label data
/// * `vectors` - Feature vectors
/// * `shuffle` - Whether to randomly shuffle the data
pub fn prepare_data_iteration(
    labels: &[String],
    vectors: &[String],
    shuffle: bool,
) -> (Vec<String>, Vec<String>) {
    if shuffle {
        let mut indices: Vec<usize> = (0..labels.len()).collect();
        indices.shuffle(&mut thread_rng());

        let label_refs: Vec<_> = indices.iter().map(|&i| labels[i].clone()).collect();
        let vector_refs: Vec<_> = indices.iter().map(|&i| vectors[i].clone()).collect();

        (label_refs, vector_refs)
    } else {
        (labels.to_vec(), vectors.to_vec())
    }
}

/// Splits data into training and validation sets
///
/// # Arguments
/// * `labels` - All labels
/// * `vectors` - All feature vectors
/// * `validation_ratio` - Fraction of data to use for validation
///
/// # Returns
/// Tuple of ((train_labels, train_vectors), (val_labels, val_vectors))
pub fn split_data<'a>(
    labels: &'a [String],
    vectors: &'a [String],
    validation_ratio: f32,
) -> ((&'a [String], &'a [String]), (&'a [String], &'a [String])) {
    let total_samples = labels.len();
    let validation_size = (total_samples as f32 * validation_ratio) as usize;

    let validation_labels = &labels[..validation_size];
    let validation_vectors = &vectors[..validation_size];
    let train_labels = &labels[validation_size..];
    let train_vectors = &vectors[validation_size..];

    (
        (train_labels, train_vectors),
        (validation_labels, validation_vectors),
    )
}
