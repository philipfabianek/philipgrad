#[cfg(test)]
use crate::Tensor;

/// Asserts that two floating point values are approximately equal
///
/// # Arguments
/// * `a` - First value
/// * `b` - Second value
/// * `epsilon` - Maximum allowed difference
#[cfg(test)]
pub fn assert_close(a: f32, b: f32, epsilon: f32) {
    assert!((a - b).abs() <= epsilon);
}

/// Asserts that two tensors are exactly equal in both shape and values
///
/// # Arguments
/// * `a` - First tensor
/// * `b` - Second tensor
#[cfg(test)]
pub fn assert_tensors_eq(a: &Tensor, b: &Tensor) {
    assert_eq!(a.shape, b.shape);
    for (x, y) in a.data.iter().zip(b.data.iter()) {
        assert_eq!(x, y)
    }
}
