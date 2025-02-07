use crate::{LayerBackwardContext, LayerForwardContext, Tensor};

/// Implements 2D max pooling layer.
/// Reduces spatial dimensions by taking maximum values over local regions.
#[derive(Debug)]
pub struct MaxPool {
    /// Number of input channels
    pub in_channels: usize,
    /// Size of the pooling window
    pub kernel_size: usize,
    /// Step size between pooling windows
    pub stride: usize,
}

#[derive(Debug)]
pub struct MaxPoolForwardContext {
    pub input: Tensor,
    /// Indices of maximum values for each pooling window
    pub max_indices: Vec<usize>,
}

#[derive(Debug)]
pub struct MaxPoolBackwardContext {}

impl MaxPool {
    pub fn new(in_channels: usize, kernel_size: usize, stride: usize) -> Self {
        MaxPool {
            in_channels,
            kernel_size,
            stride,
        }
    }

    pub fn forward(&self, input: Tensor, _training: bool) -> (Tensor, LayerForwardContext) {
        let (batch_size, height_width) = input.shape;
        let height = height_width / self.in_channels;
        let spatial_dim = (height as f32).sqrt() as usize;

        let out_height = (spatial_dim - self.kernel_size) / self.stride + 1;
        let out_spatial = self.in_channels * out_height * out_height;

        let mut output_data = Vec::with_capacity(batch_size * out_spatial);
        let mut max_indices = Vec::with_capacity(batch_size * out_spatial);

        // Process each batch and channel
        for b in 0..batch_size {
            for c in 0..self.in_channels {
                // Slide pooling window over height and width
                for i in 0..out_height {
                    for j in 0..out_height {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;

                        // Find maximum value in current pooling window
                        for ki in 0..self.kernel_size {
                            let h = i * self.stride + ki;
                            let idx_offset =
                                b * height_width + c * spatial_dim * spatial_dim + h * spatial_dim;

                            for kj in 0..self.kernel_size {
                                let w = j * self.stride + kj;

                                let idx = idx_offset + w;
                                let val = input.data[idx];

                                if val > max_val {
                                    max_val = val;
                                    max_idx = idx;
                                }
                            }
                        }

                        output_data.push(max_val);
                        max_indices.push(max_idx);
                    }
                }
            }
        }

        let output = Tensor::new_with_shape(output_data, (batch_size, out_spatial));
        (
            output,
            LayerForwardContext::MaxPool(MaxPoolForwardContext { input, max_indices }),
        )
    }

    pub fn backward(
        &self,
        grad_output: Tensor,
        context: &MaxPoolForwardContext,
    ) -> (Tensor, LayerBackwardContext) {
        let (batch_size, height_width) = context.input.shape;
        let mut grad_input = vec![0.0; height_width * batch_size];

        // Route gradients back through max indices
        for b in 0..batch_size {
            for i in 0..grad_output.shape.1 {
                let out_idx = b * grad_output.shape.1 + i;
                let max_idx = context.max_indices[out_idx];
                grad_input[max_idx] += grad_output.data[out_idx];
            }
        }

        let grad_input = Tensor::new_with_shape(grad_input, (batch_size, height_width));
        (
            grad_input,
            LayerBackwardContext::MaxPool(MaxPoolBackwardContext {}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_tensors_eq;

    #[test]
    fn test_maxpool_forward_1() {
        // Single image, 4x4 input, 2x2 pool
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
            ],
            (1, 16),
        );

        let maxpool = MaxPool::new(1, 2, 2);
        let (output, _) = maxpool.forward(input, false);

        assert_eq!(output.shape, (1, 4));

        let expected = Tensor::new_with_shape(vec![6.0, 8.0, 14.0, 16.0], (1, 4));
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_maxpool_forward_2() {
        // Two images, 4x4 input, 2x2 pool
        let input = Tensor::new_with_shape(
            vec![
                // First image
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
                // Second image
                2.0, 3.0, 4.0, 1.0, //
                6.0, 7.0, 8.0, 5.0, //
                10.0, 11.0, 12.0, 9.0, //
                14.0, 15.0, 16.0, 13.0, //
            ],
            (2, 16),
        );

        let maxpool = MaxPool::new(1, 2, 2);
        let (output, _) = maxpool.forward(input, false);

        assert_eq!(output.shape, (2, 4));

        let expected = Tensor::new_with_shape(
            vec![
                6.0, 8.0, 14.0, 16.0, //
                7.0, 8.0, 15.0, 16.0, //
            ],
            (2, 4),
        );
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_maxpool_forward_3() {
        // One image, two channels, 4x4 input, 2x2 pool
        let input = Tensor::new_with_shape(
            vec![
                // First channel
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
                // Second channel
                2.0, 3.0, 4.0, 1.0, //
                6.0, 7.0, 8.0, 5.0, //
                10.0, 11.0, 12.0, 9.0, //
                14.0, 15.0, 16.0, 13.0, //
            ],
            (1, 32),
        );

        let maxpool = MaxPool::new(2, 2, 2);
        let (output, _) = maxpool.forward(input, false);

        assert_eq!(output.shape, (1, 8));

        let expected = Tensor::new_with_shape(
            vec![
                6.0, 8.0, 14.0, 16.0, //
                7.0, 8.0, 15.0, 16.0, //
            ],
            (1, 8),
        );
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_maxpool_forward_4() {
        // Two images, two channels, 4x4 input, 2x2 pool
        let input = Tensor::new_with_shape(
            vec![
                // First image, first channel
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
                // First image, second channel
                2.0, 3.0, 4.0, 1.0, //
                6.0, 7.0, 8.0, 5.0, //
                10.0, 11.0, 12.0, 9.0, //
                14.0, 15.0, 16.0, 13.0, //
                // Second image, first channel
                1.0, 2.0, 3.0, 4.0, //
                5.0, 11.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
                // Second image, second channel
                2.0, 3.0, 4.0, 1.0, //
                6.0, 7.0, 8.0, 5.0, //
                10.0, 11.0, 19.0, 9.0, //
                14.0, 15.0, 16.0, 13.0, //
            ],
            (2, 32),
        );

        let maxpool = MaxPool::new(2, 2, 2);
        let (output, _) = maxpool.forward(input, false);

        assert_eq!(output.shape, (2, 8));

        let expected = Tensor::new_with_shape(
            vec![
                6.0, 8.0, 14.0, 16.0, //
                7.0, 8.0, 15.0, 16.0, //
                11.0, 8.0, 14.0, 16.0, //
                7.0, 8.0, 15.0, 19.0, //
            ],
            (2, 8),
        );
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_maxpool_forward_5() {
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
            ],
            (1, 9),
        );

        let maxpool = MaxPool::new(1, 2, 2);
        let (output, _) = maxpool.forward(input, false);

        assert_eq!(output.shape, (1, 1));

        let expected = Tensor::new_with_shape(vec![5.0], (1, 1));
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_maxpool_backward_1() {
        // Single image, 4x4 input, 2x2 pool
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
            ],
            (1, 16),
        );

        let maxpool = MaxPool::new(1, 2, 2);
        let (_, context) = maxpool.forward(input.clone(), false);

        let context = match context {
            LayerForwardContext::MaxPool(context) => context,
            _ => panic!(),
        };

        let grad_output = Tensor::new_with_shape(
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
            ],
            (1, 4),
        );

        let (grad_input, _) = maxpool.backward(grad_output, &context);

        assert_eq!(grad_input.shape, (1, 16));

        let expected_grad_input = Tensor::new_with_shape(
            vec![
                0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 2.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 3.0, 0.0, 4.0, //
            ],
            (1, 16),
        );

        assert_tensors_eq(&grad_input, &expected_grad_input);
    }

    #[test]
    fn test_maxpool_backward_3() {
        // Two channels, 4x4 input, 2x2 pool
        let input = Tensor::new_with_shape(
            vec![
                // First channel
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
                // Second channel
                2.0, 3.0, 4.0, 1.0, //
                6.0, 7.0, 8.0, 5.0, //
                10.0, 11.0, 12.0, 9.0, //
                14.0, 15.0, 16.0, 13.0, //
            ],
            (1, 32),
        );

        let maxpool = MaxPool::new(2, 2, 2);
        let (_, context) = maxpool.forward(input.clone(), false);

        let context = match context {
            LayerForwardContext::MaxPool(context) => context,
            _ => panic!(),
        };

        let grad_output = Tensor::new_with_shape(
            vec![
                // First channel
                1.0, 2.0, //
                3.0, 4.0, //
                // Second channel
                5.0, 6.0, //
                7.0, 8.0, //
            ],
            (1, 8),
        );

        let (grad_input, _) = maxpool.backward(grad_output, &context);

        assert_eq!(grad_input.shape, (1, 32));

        let expected_grad_input = Tensor::new_with_shape(
            vec![
                // First channel
                0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 2.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 3.0, 0.0, 4.0, //
                // Second channel
                0.0, 0.0, 0.0, 0.0, //
                0.0, 5.0, 6.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 7.0, 8.0, 0.0, //
            ],
            (1, 32),
        );

        assert_tensors_eq(&grad_input, &expected_grad_input);
    }

    #[test]
    fn test_maxpool_backward_4() {
        // Two images, two channels, 4x4 input, 2x2 pool
        let input = Tensor::new_with_shape(
            vec![
                // First image, first channel
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
                // First image, second channel
                2.0, 3.0, 4.0, 1.0, //
                6.0, 7.0, 8.0, 5.0, //
                10.0, 11.0, 12.0, 9.0, //
                14.0, 15.0, 16.0, 13.0, //
                // Second image, first channel
                1.0, 2.0, 3.0, 4.0, //
                5.0, 11.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
                // Second image, second channel
                2.0, 3.0, 4.0, 1.0, //
                6.0, 7.0, 8.0, 5.0, //
                10.0, 11.0, 19.0, 9.0, //
                14.0, 15.0, 16.0, 13.0, //
            ],
            (2, 32),
        );

        let maxpool = MaxPool::new(2, 2, 2);
        let (_, context) = maxpool.forward(input.clone(), false);

        let context = match context {
            LayerForwardContext::MaxPool(context) => context,
            _ => panic!(),
        };

        let grad_output = Tensor::new_with_shape(
            vec![
                // First image, first channel
                1.0, 2.0, //
                3.0, 4.0, //
                // First image, second channel
                5.0, 6.0, //
                7.0, 8.0, //
                // Second image, first channel
                9.0, 10.0, //
                11.0, 12.0, //
                // Second image, second channel
                13.0, 14.0, //
                15.0, 16.0, //
            ],
            (2, 8),
        );

        let (grad_input, _) = maxpool.backward(grad_output, &context);

        assert_eq!(grad_input.shape, (2, 32));

        let expected_grad_input = Tensor::new_with_shape(
            vec![
                // First image, first channel
                0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 2.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 3.0, 0.0, 4.0, //
                // First image, second channel
                0.0, 0.0, 0.0, 0.0, //
                0.0, 5.0, 6.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 7.0, 8.0, 0.0, //
                // Second image, first channel
                0.0, 0.0, 0.0, 0.0, //
                0.0, 9.0, 0.0, 10.0, //
                0.0, 0.0, 0.0, 0.0, //
                0.0, 11.0, 0.0, 12.0, //
                // Second image, second channel
                0.0, 0.0, 0.0, 0.0, //
                0.0, 13.0, 14.0, 0.0, //
                0.0, 0.0, 16.0, 0.0, //
                0.0, 15.0, 0.0, 0.0, //
            ],
            (2, 32),
        );

        assert_tensors_eq(&grad_input, &expected_grad_input);
    }
}
