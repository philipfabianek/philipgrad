/// Parses a comma-separated string of numbers into a vector of f32,
/// uses unsafe optimization for performance in parsing large datasets
pub fn parse_vector_str(vector_str: &str) -> Vec<f32> {
    let values_count = vector_str.as_bytes().iter().filter(|&&b| b == b',').count() + 1;
    let mut values = Vec::with_capacity(values_count);

    // Optimized parsing
    vector_str
        .as_bytes()
        .split(|&b| b == b',')
        .for_each(|slice| {
            let num = unsafe { std::str::from_utf8_unchecked(slice).parse::<f32>().unwrap() };
            values.push(num);
        });

    values
}

/// Parses a single sample consisting of feature vector and label
pub fn parse_sample(vector_str: &str, label_str: &str) -> (Vec<f32>, f32) {
    let values = parse_vector_str(vector_str.trim());
    let target = label_str.trim().parse::<f32>().unwrap();
    (values, target)
}
