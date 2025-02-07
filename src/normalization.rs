use crate::parse_vector_str;

/// Calculates mean and standard deviation for a dataset,
/// internally uses f64 for accuracy reasons, f32 did not work correctly
///
/// # Arguments
/// * `lines` - Vector of strings containing numerical data
///
/// # Returns
/// Tuple of (mean, standard_deviation) as f32
fn calculate_average_and_standard_deviation(lines: &Vec<String>) -> (f32, f32) {
    let mut all_values = Vec::new();

    // Parse all values into a single vector
    for line in lines {
        let values = parse_vector_str(line);
        all_values.extend(values.into_iter().map(|x| x as f64));
    }

    let count = all_values.len() as f64;

    // Calculate mean
    let mean: f64 = all_values.iter().sum::<f64>() / count;

    // Calculate variance and standard deviation
    let variance = all_values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / count;

    let std_dev = variance.sqrt();

    (mean as f32, std_dev as f32)
}

/// Parameters for feature normalization
#[derive(Debug)]
pub struct NormalizationParams {
    /// Mean of the dataset
    pub mean: f32,
    /// Standard deviation of the dataset
    pub std_dev: f32,
}

impl NormalizationParams {
    /// Creates normalization parameters from input data
    pub fn from_data(vectors: &Vec<String>) -> Self {
        let (mean, std_dev) = calculate_average_and_standard_deviation(vectors);
        Self { mean, std_dev }
    }
}
