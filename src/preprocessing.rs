/// Creates one-hot encoded targets with optional label smoothing
///
/// # Arguments
/// * `targets` - Vector of class indices
/// * `smoothing` - Label smoothing factor (0 = no smoothing)
pub fn create_one_hot(targets: &[f32], smoothing: f32) -> Vec<f32> {
    let num_classes = 10;
    let smooth_value = smoothing / (num_classes - 1) as f32;

    let mut one_hot = Vec::with_capacity(targets.len() * num_classes);
    for (_, &target) in targets.iter().enumerate() {
        for j in 0..num_classes {
            one_hot.push(if j == target as usize {
                1.0 - smoothing
            } else {
                smooth_value
            });
        }
    }
    one_hot
}
