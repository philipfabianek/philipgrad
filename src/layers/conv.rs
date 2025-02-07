use crate::{LayerBackwardContext, LayerForwardContext, Tensor};
use rand::distributions::Distribution;
use rand::thread_rng;
use rand_distr::Normal;

/// Implements a 2D convolutional layer.
#[derive(Debug)]
pub struct Conv {
    /// Number of input channels
    pub in_channels: usize,
    /// Kernel dimensions (height, width)
    pub kernel: (usize, usize),
    /// Stride in both dimensions
    pub stride: (usize, usize),
    /// Zero padding in both dimensions
    pub padding: (usize, usize),
    /// Convolutional filters
    pub weights: Tensor,
    /// Bias terms for each output channel
    pub bias: Tensor,
    /// Accumulated weight gradients
    pub grad_weights: Tensor,
    /// Accumulated bias gradients
    pub grad_bias: Tensor,
}

#[derive(Debug)]
pub struct ConvForwardContext {
    pub input_shape: (usize, usize),
    /// Im2row transformed input for efficient computation
    pub cols: Tensor,
}

#[derive(Debug)]
pub struct ConvBackwardContext {
    pub grad_weights: Tensor,
    pub grad_bias: Tensor,
}

impl Conv {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self::new_with_alpha(in_channels, out_channels, kernel, stride, padding, 0.0)
    }

    /// Creates a new Conv layer with He initialization scaled for LeakyReLU
    pub fn new_with_alpha(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        leaky_relu_alpha: f32,
    ) -> Self {
        let mut rng = thread_rng();
        let (kernel_h, kernel_w) = kernel;

        // He initialization scaled for LeakyReLU
        let n = in_channels * kernel_h * kernel_w;
        let weight_scale = (2.0 / ((1.0 + leaky_relu_alpha.powi(2)) * (n as f32))).sqrt();
        let normal = Normal::new(0.0, weight_scale).unwrap();

        let weights = (0..out_channels * in_channels * kernel_h * kernel_w)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let weights =
            Tensor::new_with_shape(weights, (out_channels, in_channels * kernel_h * kernel_w));

        let bias = Tensor::new_with_shape(vec![0.0; out_channels], (out_channels, 1));

        let grad_weights = Tensor::zeros_like(&weights);
        let grad_bias = Tensor::zeros_like(&bias);

        Conv {
            in_channels,
            kernel,
            stride,
            padding,
            weights,
            bias,
            grad_weights,
            grad_bias,
        }
    }

    pub fn forward(&self, input: Tensor, _training: bool) -> (Tensor, LayerForwardContext) {
        let input_shape = input.shape;
        let (batch_size, _in_channels, height, width) =
            Tensor::shape4(input.shape, self.in_channels);
        let (kernel_h, kernel_w) = self.kernel;

        // Calculate output dimensions after convolution
        let out_height = (height + 2 * self.padding.0 - kernel_h) / self.stride.0 + 1;
        let out_width = (width + 2 * self.padding.1 - kernel_w) / self.stride.1 + 1;

        // Apply padding if needed
        let input_padded = if self.padding != (0, 0) {
            input.pad2d(self.in_channels, self.padding)
        } else {
            input
        };

        // Transform input for efficient convolution computation
        let cols = input_padded.im2row(self.in_channels, kernel_h, kernel_w, self.stride);
        let mut output = self.weights.matmul(&cols.transpose());

        // Rearrange output to group channels by batch
        let mut rearranged_output = Vec::new();
        let cc = output.shape.1 / batch_size;
        for b in 0..batch_size {
            for r in 0..output.shape.0 {
                for c in 0..cc {
                    let col_idx = b * cc + c;
                    rearranged_output.push(output.data[r * output.shape.1 + col_idx]);
                }
            }
        }

        let out_shape = (batch_size, self.weights.shape.0 * out_height * out_width);
        output = Tensor::new_with_shape(rearranged_output, out_shape);

        // Add bias terms
        for i in 0..batch_size {
            for c in 0..self.weights.shape.0 {
                let bias_val = self.bias.data[c];
                for hw in 0..out_height * out_width {
                    let idx = (i * self.weights.shape.0 + c) * (out_height * out_width) + hw;
                    output.data[idx] += bias_val;
                }
            }
        }

        (
            output,
            LayerForwardContext::Conv(ConvForwardContext { input_shape, cols }),
        )
    }

    pub fn backward(
        &self,
        grad_output: Tensor,
        context: &ConvForwardContext,
    ) -> (Tensor, LayerBackwardContext) {
        let (batch_size, in_channels, height, width) =
            Tensor::shape4(context.input_shape, self.in_channels);

        let (kernel_h, kernel_w) = self.kernel;
        let out_height = (height + 2 * self.padding.0 - kernel_h) / self.stride.0 + 1;
        let out_width = (width + 2 * self.padding.1 - kernel_w) / self.stride.1 + 1;

        // Rearrange gradients to match forward pass format
        let mut rearranged_grad = Vec::with_capacity(grad_output.data.len());
        let out_spatial = out_height * out_width;

        for c in 0..self.weights.shape.0 {
            for b in 0..batch_size {
                for hw in 0..out_spatial {
                    let idx = (b * self.weights.shape.0 + c) * out_spatial + hw;
                    rearranged_grad.push(grad_output.data[idx]);
                }
            }
        }

        // Compute bias gradients
        let mut grad_bias = vec![0.0; self.weights.shape.0];
        for c in 0..self.weights.shape.0 {
            for b in 0..batch_size {
                for hw in 0..out_spatial {
                    let idx = c * (batch_size * out_spatial) + (b * out_spatial) + hw;
                    grad_bias[c] += rearranged_grad[idx];
                }
            }
        }
        let grad_bias = Tensor::new_with_shape(grad_bias, (self.weights.shape.0, 1));

        let grad_output = Tensor::new_with_shape(
            rearranged_grad,
            (self.weights.shape.0, batch_size * out_height * out_width),
        );

        // Compute weight and input gradients
        let grad_weights = grad_output.matmul(&context.cols);
        let grad_cols = self.weights.transpose().matmul(&grad_output);
        let grad_input = grad_cols.transpose().row2im(
            self.in_channels,
            (batch_size, in_channels * height * width),
            kernel_h,
            kernel_w,
            self.stride,
            self.padding,
        );

        (
            grad_input,
            LayerBackwardContext::Conv(ConvBackwardContext {
                grad_weights,
                grad_bias,
            }),
        )
    }

    pub fn update_grads(&mut self, context: &ConvBackwardContext) {
        for (dest, src) in self
            .grad_weights
            .data
            .iter_mut()
            .zip(context.grad_weights.data.iter())
        {
            *dest += src;
        }

        for (dest, src) in self
            .grad_bias
            .data
            .iter_mut()
            .zip(context.grad_bias.data.iter())
        {
            *dest += src;
        }
    }

    pub fn get_parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights, &self.bias]
    }

    pub fn get_parameter_pairs(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        vec![
            (&mut self.weights, &self.grad_weights),
            (&mut self.bias, &self.grad_bias),
        ]
    }

    pub fn clear_grads(&mut self) {
        for val in self.grad_weights.data.iter_mut() {
            *val = 0.0;
        }
        for val in self.grad_bias.data.iter_mut() {
            *val = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_tensors_eq;

    #[test]
    fn test_conv_forward_1() {
        // Single image, single channel, 4x4 input, 2x2 kernel, single output channel
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                8.0, 7.0, 6.0, 5.0, //
                4.0, 3.0, 2.0, 1.0, //
            ],
            (1, 16),
        );

        let mut conv = Conv::new(1, 1, (2, 2), (1, 1), (0, 0));

        conv.weights = Tensor::new_with_shape(
            vec![
                1.0, -1.0, //
                1.0, -1.0, //
            ],
            (1, 4),
        );
        conv.bias = Tensor::new_with_shape(vec![1.0], (1, 1));

        let (output, _) = conv.forward(input, false);

        assert_eq!(output.shape, (1, 9));

        let expected = Tensor::new_with_shape(
            vec![
                -1.0, -1.0, -1.0, //
                1.0, 1.0, 1.0, //
                3.0, 3.0, 3.0, //
            ],
            (1, 9),
        );
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_conv_forward_2() {
        // Two images, single channel, 4x4 input, 2x2 kernel, single output channel
        let input = Tensor::new_with_shape(
            vec![
                // First image
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
                // Second image
                17.0, 18.0, 19.0, 20.0, //
                21.0, 22.0, 23.0, 24.0, //
                25.0, 26.0, 27.0, 28.0, //
                29.0, 30.0, 31.0, 32.0, //
            ],
            (2, 16),
        );

        let mut conv = Conv::new(1, 1, (2, 2), (1, 1), (0, 0));

        conv.weights = Tensor::new_with_shape(
            vec![
                1.0, 1.0, //
                1.0, 1.0, //
            ],
            (1, 4),
        );
        conv.bias = Tensor::new_with_shape(vec![1.0], (1, 1));

        let (output, _) = conv.forward(input, false);

        assert_eq!(output.shape, (2, 9));

        let expected = Tensor::new_with_shape(
            vec![
                // First image
                15.0, 19.0, 23.0, //
                31.0, 35.0, 39.0, //
                47.0, 51.0, 55.0, //
                // Second image
                79.0, 83.0, 87.0, //
                95.0, 99.0, 103.0, //
                111.0, 115.0, 119.0, //
            ],
            (2, 9),
        );
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_conv_forward_3() {
        // Single image, two channels, 4x4 input, 2x2 kernel, single output channel
        let input = Tensor::new_with_shape(
            vec![
                // First channel
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
                // Second channel
                16.0, 15.0, 14.0, 13.0, //
                12.0, 11.0, 10.0, 9.0, //
                8.0, 7.0, 6.0, 5.0, //
                4.0, 3.0, 2.0, 1.0, //
            ],
            (1, 32),
        );

        let mut conv = Conv::new(2, 1, (2, 2), (1, 1), (0, 0));

        conv.weights = Tensor::new_with_shape(
            vec![
                1.0, 1.0, //
                1.0, 1.0, //
                ////////////
                -1.0, -1.0, //
                -1.0, -1.0, //
            ],
            (1, 8),
        );
        conv.bias = Tensor::new_with_shape(vec![1.0], (1, 1));

        let (output, _) = conv.forward(input, false);

        assert_eq!(output.shape, (1, 9));

        let expected = Tensor::new_with_shape(
            vec![
                -39.0, -31.0, -23.0, //
                -7.0, 1.0, 9.0, //
                25.0, 33.0, 41.0, //
            ],
            (1, 9),
        );
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_conv_forward_4() {
        // Single image, single input channel, 4x4 input, 2x2 kernel, two output channels
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
            ],
            (1, 16),
        );

        let mut conv = Conv::new(1, 2, (2, 2), (1, 1), (0, 0));

        conv.weights = Tensor::new_with_shape(
            vec![
                // First output channel
                1.0, 1.0, //
                1.0, 1.0, //
                // Second output channel
                -1.0, -1.0, //
                -1.0, -1.0, //
            ],
            (2, 4),
        );
        conv.bias = Tensor::new_with_shape(vec![1.0, 0.0], (2, 1));

        let (output, _) = conv.forward(input, false);

        assert_eq!(output.shape, (1, 18));

        let expected = Tensor::new_with_shape(
            vec![
                // First output channel
                15.0, 19.0, 23.0, //
                31.0, 35.0, 39.0, //
                47.0, 51.0, 55.0, //
                // Second output channel
                -14.0, -18.0, -22.0, //
                -30.0, -34.0, -38.0, //
                -46.0, -50.0, -54.0, //
            ],
            (1, 18),
        );
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_conv_forward_5() {
        // Two images, single input channel, 4x4 input, 2x2 kernel, TWO output channels
        let input = Tensor::new_with_shape(
            vec![
                // First image
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
                // Second image
                2.0, 3.0, 4.0, 5.0, //
                6.0, 7.0, 8.0, 9.0, //
                10.0, 11.0, 12.0, 13.0, //
                14.0, 15.0, 16.0, 17.0, //
            ],
            (2, 16),
        );

        let mut conv = Conv::new(1, 2, (2, 2), (1, 1), (0, 0));

        conv.weights = Tensor::new_with_shape(
            vec![
                // First output channel
                1.0, 1.0, //
                1.0, 1.0, //
                // Second output channel
                2.0, 2.0, //
                2.0, 2.0, //
            ],
            (2, 4),
        );
        conv.bias = Tensor::new_with_shape(vec![1.0, -1.0], (2, 1));

        let (output, _) = conv.forward(input, false);

        assert_eq!(output.shape, (2, 18));

        let expected = Tensor::new_with_shape(
            vec![
                // First image, first output channel
                15.0, 19.0, 23.0, //
                31.0, 35.0, 39.0, //
                47.0, 51.0, 55.0, //
                // First image, second output channel
                27.0, 35.0, 43.0, //
                59.0, 67.0, 75.0, //
                91.0, 99.0, 107.0, //
                // Second image, first output channel
                19.0, 23.0, 27.0, //
                35.0, 39.0, 43.0, //
                51.0, 55.0, 59.0, //
                // Second image, second output channel
                35.0, 43.0, 51.0, //
                67.0, 75.0, 83.0, //
                99.0, 107.0, 115.0, //
            ],
            (2, 18),
        );
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_conv_forward_6() {
        // Single image, two input channels, 4x4 input, 2x2 kernel, two output channels
        let input = Tensor::new_with_shape(
            vec![
                // First channel
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
                // Second channel
                16.0, 15.0, 14.0, 13.0, //
                12.0, 11.0, 10.0, 9.0, //
                8.0, 7.0, 6.0, 5.0, //
                4.0, 3.0, 2.0, 1.0, //
            ],
            (1, 32),
        );

        let mut conv = Conv::new(2, 2, (2, 2), (1, 1), (0, 0));

        conv.weights = Tensor::new_with_shape(
            vec![
                // First output channel
                1.0, 1.0, //
                1.0, 1.0, //
                ////////////
                -1.0, -1.0, //
                -1.0, -1.0, //
                // Second output channel
                -1.0, -1.0, //
                -1.0, -1.0, //
                //////////////
                1.0, 1.0, //
                1.0, 1.0, //
            ],
            (2, 8),
        );
        conv.bias = Tensor::new_with_shape(vec![1.0, -1.0], (2, 1));

        let (output, _) = conv.forward(input, false);

        assert_eq!(output.shape, (1, 18));

        let expected = Tensor::new_with_shape(
            vec![
                // First output channel
                -39.0, -31.0, -23.0, //
                -7.0, 1.0, 9.0, //
                25.0, 33.0, 41.0, //
                // Second output channel
                39.0, 31.0, 23.0, //
                7.0, -1.0, -9.0, //
                -25.0, -33.0, -41.0, //
            ],
            (1, 18),
        );
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_conv_forward_7() {
        // Two images, two input channels, 3x3 input, 2x2 kernel, two output channels
        let input = Tensor::new_with_shape(
            vec![
                // First image, first channel
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
                // First image, second channel
                9.0, 8.0, 7.0, //
                6.0, 5.0, 4.0, //
                3.0, 2.0, 1.0, //
                // Second image, first channel
                1.0, 3.0, 5.0, //
                7.0, -9.0, 11.0, //
                -1.0, -2.0, 3.0, //
                // Second image, second channel
                0.0, 12.0, -5.0, //
                7.0, 9.0, 15.0, //
                -13.0, 11.0, -1.0, //
            ],
            (2, 18),
        );

        let mut conv = Conv::new(2, 2, (2, 2), (1, 1), (0, 0));

        conv.weights = Tensor::new_with_shape(
            vec![
                // First output channel
                1.0, 1.0, //
                1.0, 1.0, //
                ////////////
                -1.0, -1.0, //
                -1.0, -1.0, //
                // Second output channel
                -1.0, 0.0, //
                -1.0, -1.0, //
                ////////////
                0.0, 1.0, //
                1.0, 1.0, //
            ],
            (2, 8),
        );
        conv.bias = Tensor::new_with_shape(vec![1.0, -1.0], (2, 1));

        let (output, _) = conv.forward(input, false);

        assert_eq!(output.shape, (2, 8));

        let expected = Tensor::new_with_shape(
            vec![
                // First image, first output channel
                -15.0, -7.0, //
                9.0, 17.0, //
                // First image, second output channel
                8.0, 2.0, //
                -10.0, -16.0, //
                // Second image, first output channel
                -25.0, -20.0, //
                -18.0, -30.0, //
                // Second image, second output channel
                28.0, 13.0, //
                2.0, 32.0, //
            ],
            (2, 8),
        );
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_conv_backward_1() {
        // Single image, single channel, 4x4 input, 2x2 kernel, single output channel
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                8.0, 7.0, 6.0, 5.0, //
                4.0, 3.0, 2.0, 1.0, //
            ],
            (1, 16),
        );

        let mut conv = Conv::new(1, 1, (2, 2), (1, 1), (0, 0));

        conv.weights = Tensor::new_with_shape(
            vec![
                1.0, -2.0, //
                2.0, -3.0, //
            ],
            (1, 4),
        );
        conv.bias = Tensor::new_with_shape(vec![1.0], (1, 1));

        let (_output, context) = conv.forward(input.clone(), false);

        let context = match context {
            LayerForwardContext::Conv(context) => context,
            _ => panic!(),
        };

        let grad_output = Tensor::new_with_shape(
            vec![
                2.0, 1.0, -1.0, //
                -1.0, 0.0, 1.0, //
                0.0, 1.0, -2.0, //
            ],
            (1, 9),
        );

        let (grad_input, backward_context) = conv.backward(grad_output, &context);

        let ConvBackwardContext {
            grad_weights,
            grad_bias,
        } = match backward_context {
            LayerBackwardContext::Conv(ctx) => ctx,
            _ => panic!(),
        };

        assert_eq!(grad_input.shape, (1, 16));

        let expected_grad_input = Tensor::new_with_shape(
            vec![
                2.0, -3.0, -3.0, 2.0, //
                3.0, -2.0, -4.0, 1.0, //
                -2.0, 4.0, -2.0, 1.0, //
                0.0, 2.0, -7.0, 6.0, //
            ],
            (1, 16),
        );

        let expected_grad_weights = Tensor::new_with_shape(
            vec![
                -2.0, 1.0, //
                6.0, 9.0, //
            ],
            (1, 4),
        );

        let expected_grad_bias = Tensor::new_with_shape(vec![1.0], (1, 1));

        assert_tensors_eq(&grad_input, &expected_grad_input);
        assert_tensors_eq(&grad_weights, &expected_grad_weights);
        assert_tensors_eq(&grad_bias, &expected_grad_bias);
    }

    #[test]
    fn test_conv_backward_2() {
        // Two images, single input channel, 4x4 input, 2x2 kernel, single output channel
        let input = Tensor::new_with_shape(
            vec![
                // First image
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
                // Second image
                17.0, 18.0, 19.0, 20.0, //
                21.0, 22.0, 23.0, 24.0, //
                25.0, 26.0, 27.0, 28.0, //
                29.0, 30.0, 31.0, 32.0, //
            ],
            (2, 16),
        );

        let mut conv = Conv::new(1, 1, (2, 2), (1, 1), (0, 0));
        conv.weights = Tensor::new_with_shape(
            vec![
                2.0, 1.0, //
                1.0, -2.0, //
            ],
            (1, 4),
        );

        let (_output, context) = conv.forward(input.clone(), false);

        let context = match context {
            LayerForwardContext::Conv(context) => context,
            _ => panic!(),
        };

        let grad_output = Tensor::new_with_shape(
            vec![
                2.0, -1.0, 1.0, //
                1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, //
                /////////////////
                1.0, 1.0, 1.0, //
                1.0, 3.0, 1.0, //
                1.0, 1.0, 1.0, //
            ],
            (2, 9),
        );

        let (grad_input, backward_context) = conv.backward(grad_output, &context);
        let ConvBackwardContext {
            grad_weights,
            grad_bias,
        } = match backward_context {
            LayerBackwardContext::Conv(ctx) => ctx,
            _ => panic!(),
        };

        assert_eq!(grad_input.shape, (2, 16));
        assert_eq!(grad_weights.shape, (1, 4));
        assert_eq!(grad_bias.shape, (1, 1));

        let expected_grad_input = Tensor::new_with_shape(
            vec![
                // First image
                4.0, 0.0, 1.0, 1.0, //
                4.0, -2.0, 6.0, -1.0, //
                3.0, 2.0, 2.0, -1.0, //
                1.0, -1.0, -1.0, -2.0, //
                // Second image
                2.0, 3.0, 3.0, 1.0, //
                3.0, 6.0, 4.0, -1.0, //
                3.0, 4.0, -2.0, -1.0, //
                1.0, -1.0, -1.0, -2.0, //
            ],
            (2, 16),
        );

        let expected_grad_weights = Tensor::new_with_shape(
            vec![
                293.0, 312.0, //
                369.0, 388.0, //
            ],
            (1, 4),
        );

        let expected_grad_bias = Tensor::new_with_shape(vec![19.0], (1, 1));

        assert_tensors_eq(&grad_input, &expected_grad_input);
        assert_tensors_eq(&grad_weights, &expected_grad_weights);
        assert_tensors_eq(&grad_bias, &expected_grad_bias);
    }

    #[test]
    fn test_conv_backward_3() {
        // Single image, two input channels, 4x4 input, 2x2 kernel, single output channel
        let input = Tensor::new_with_shape(
            vec![
                // First channel
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                8.0, 7.0, 6.0, 5.0, //
                4.0, 3.0, 2.0, 1.0, //
                // Second channel
                4.0, 3.0, 2.0, 1.0, //
                8.0, 7.0, 6.0, 5.0, //
                5.0, 6.0, 7.0, 8.0, //
                1.0, 2.0, 3.0, 4.0, //
            ],
            (1, 32),
        );

        let mut conv = Conv::new(2, 1, (2, 2), (1, 1), (0, 0));
        conv.weights = Tensor::new_with_shape(
            vec![
                1.0, -2.0, //
                2.0, -3.0, //
                /////////////
                -1.0, 2.0, //
                -2.0, 3.0, //
            ],
            (1, 8),
        );

        let (_output, context) = conv.forward(input.clone(), false);

        let context = match context {
            LayerForwardContext::Conv(context) => context,
            _ => panic!(),
        };

        let grad_output = Tensor::new_with_shape(
            vec![
                2.0, 1.0, -1.0, //
                -1.0, 0.0, 1.0, //
                0.0, 1.0, -2.0, //
            ],
            (1, 9),
        );

        let (grad_input, backward_context) = conv.backward(grad_output, &context);
        let ConvBackwardContext {
            grad_weights,
            grad_bias,
        } = match backward_context {
            LayerBackwardContext::Conv(ctx) => ctx,
            _ => panic!(),
        };

        assert_eq!(grad_input.shape, (1, 32));
        assert_eq!(grad_weights.shape, (1, 8));
        assert_eq!(grad_bias.shape, (1, 1));

        let expected_grad_input = Tensor::new_with_shape(
            vec![
                // First channel
                2.0, -3.0, -3.0, 2.0, //
                3.0, -2.0, -4.0, 1.0, //
                -2.0, 4.0, -2.0, 1.0, //
                0.0, 2.0, -7.0, 6.0, //
                // Second channel
                -2.0, 3.0, 3.0, -2.0, //
                -3.0, 2.0, 4.0, -1.0, //
                2.0, -4.0, 2.0, -1.0, //
                0.0, -2.0, 7.0, -6.0, //
            ],
            (1, 32),
        );

        let expected_grad_weights = Tensor::new_with_shape(
            vec![
                -2.0, 1.0, //
                6.0, 9.0, //
                /////////////
                -1.0, -4.0, //
                15.0, 12.0, //
            ],
            (1, 8),
        );

        let expected_grad_bias = Tensor::new_with_shape(vec![1.0], (1, 1));

        assert_tensors_eq(&grad_input, &expected_grad_input);
        assert_tensors_eq(&grad_weights, &expected_grad_weights);
        assert_tensors_eq(&grad_bias, &expected_grad_bias);
    }

    #[test]
    fn test_conv_backward_4() {
        // Single image, single input channel, 4x4 input, 2x2 kernel, two output channels
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                8.0, 7.0, 6.0, 5.0, //
                4.0, 3.0, 2.0, 1.0, //
            ],
            (1, 16),
        );

        let mut conv = Conv::new(1, 2, (2, 2), (1, 1), (0, 0));
        conv.weights = Tensor::new_with_shape(
            vec![
                1.0, -2.0, //
                2.0, -3.0, //
                /////////////
                -1.0, 2.0, //
                -2.0, 3.0, //
            ],
            (2, 4),
        );

        let (_output, context) = conv.forward(input.clone(), false);

        let context = match context {
            LayerForwardContext::Conv(context) => context,
            _ => panic!(),
        };

        let grad_output = Tensor::new_with_shape(
            vec![
                // First output channel
                2.0, 1.0, -1.0, //
                -1.0, 0.0, 1.0, //
                0.0, 1.0, -2.0, //
                // Second output channel
                -2.0, -1.0, 1.0, //
                1.0, 0.0, -1.0, //
                0.0, -1.0, 2.0, //
            ],
            (1, 18),
        );

        let (grad_input, backward_context) = conv.backward(grad_output, &context);
        let ConvBackwardContext {
            grad_weights,
            grad_bias,
        } = match backward_context {
            LayerBackwardContext::Conv(ctx) => ctx,
            _ => panic!(),
        };

        assert_eq!(grad_input.shape, (1, 16));
        assert_eq!(grad_weights.shape, (2, 4));
        assert_eq!(grad_bias.shape, (2, 1));

        let expected_grad_input = Tensor::new_with_shape(
            vec![
                4.0, -6.0, -6.0, 4.0, //
                6.0, -4.0, -8.0, 2.0, //
                -4.0, 8.0, -4.0, 2.0, //
                0.0, 4.0, -14.0, 12.0, //
            ],
            (1, 16),
        );

        let expected_grad_weights = Tensor::new_with_shape(
            vec![
                -2.0, 1.0, //
                6.0, 9.0, //
                /////////////
                2.0, -1.0, //
                -6.0, -9.0, //
            ],
            (2, 4),
        );

        let expected_grad_bias = Tensor::new_with_shape(vec![1.0, -1.0], (2, 1));

        assert_tensors_eq(&grad_input, &expected_grad_input);
        assert_tensors_eq(&grad_weights, &expected_grad_weights);
        assert_tensors_eq(&grad_bias, &expected_grad_bias);
    }

    #[test]
    fn test_conv_backward_5() {
        // Two images, two input channels, 4x4 input, 2x2 kernel, two output channels
        let input = Tensor::new_with_shape(
            vec![
                // First image, first channel
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                8.0, 7.0, 6.0, 5.0, //
                4.0, 3.0, 2.0, 1.0, //
                // First image, second channel
                4.0, 3.0, 2.0, 1.0, //
                8.0, 7.0, 6.0, 5.0, //
                5.0, 6.0, 7.0, 8.0, //
                1.0, 2.0, 3.0, 4.0, //
                // Second image, first channel
                17.0, 18.0, 19.0, 20.0, //
                21.0, 22.0, 23.0, 24.0, //
                25.0, 26.0, 27.0, 28.0, //
                29.0, 30.0, 31.0, 32.0, //
                // Second image, second channel
                32.0, 31.0, 30.0, 29.0, //
                28.0, 27.0, 26.0, 25.0, //
                24.0, 23.0, 22.0, 21.0, //
                20.0, 19.0, 18.0, 17.0, //
            ],
            (2, 32),
        );

        let mut conv = Conv::new(2, 2, (2, 2), (1, 1), (0, 0));
        conv.weights = Tensor::new_with_shape(
            vec![
                // First output channel
                1.0, -2.0, //
                2.0, -3.0, //
                /////////////
                -1.0, 2.0, //
                -2.0, 3.0, //
                // Second output channel
                -1.0, 2.0, //
                -2.0, 3.0, //
                /////////////
                1.0, -2.0, //
                2.0, -3.0, //
            ],
            (2, 8),
        );

        let (_output, context) = conv.forward(input.clone(), false);

        let context = match context {
            LayerForwardContext::Conv(context) => context,
            _ => panic!(),
        };

        let grad_output = Tensor::new_with_shape(
            vec![
                // First image, first output channel
                2.0, 1.0, -1.0, //
                -1.0, 0.0, 1.0, //
                0.0, 1.0, -2.0, //
                // First image, second output channel
                -2.0, -1.0, 1.0, //
                1.0, 0.0, -1.0, //
                0.0, -1.0, 2.0, //
                // Second image, first output channel
                1.0, 1.0, 1.0, //
                1.0, 3.0, 1.0, //
                1.0, 1.0, 1.0, //
                // Second image, second output channel
                -1.0, -1.0, -1.0, //
                -1.0, -3.0, -1.0, //
                -1.0, -1.0, -1.0, //
            ],
            (2, 18),
        );

        let (grad_input, backward_context) = conv.backward(grad_output, &context);
        let ConvBackwardContext {
            grad_weights,
            grad_bias,
        } = match backward_context {
            LayerBackwardContext::Conv(ctx) => ctx,
            _ => panic!(),
        };

        assert_eq!(grad_input.shape, (2, 32));
        assert_eq!(grad_weights.shape, (2, 8));
        assert_eq!(grad_bias.shape, (2, 1));

        let expected_grad_input = Tensor::new_with_shape(
            vec![
                // First image, first channel
                4.0, -6.0, -6.0, 4.0, //
                6.0, -4.0, -8.0, 2.0, //
                -4.0, 8.0, -4.0, 2.0, //
                0.0, 4.0, -14.0, 12.0, //
                // First image, second channel
                -4.0, 6.0, 6.0, -4.0, //
                -6.0, 4.0, 8.0, -2.0, //
                4.0, -8.0, 4.0, -2.0, //
                0.0, -4.0, 14.0, -12.0, //
                // Second image, first channel
                2.0, -2.0, -2.0, -4.0, //
                6.0, 0.0, -12.0, -10.0, //
                6.0, 4.0, -16.0, -10.0, //
                4.0, -2.0, -2.0, -6.0, //
                // Second image, second channel
                -2.0, 2.0, 2.0, 4.0, //
                -6.0, 0.0, 12.0, 10.0, //
                -6.0, -4.0, 16.0, 10.0, //
                -4.0, 2.0, 2.0, 6.0, //
            ],
            (2, 32),
        );

        let expected_grad_weights = Tensor::new_with_shape(
            vec![
                // First output channel weights
                240.0, 254.0, //
                292.0, 306.0, //
                /////////////
                296.0, 282.0, //
                268.0, 254.0, //
                // Second output channel weights
                -240.0, -254.0, //
                -292.0, -306.0, //
                /////////////
                -296.0, -282.0, //
                -268.0, -254.0, //
            ],
            (2, 8),
        );

        let expected_grad_bias = Tensor::new_with_shape(vec![12.0, -12.0], (2, 1));

        assert_tensors_eq(&grad_input, &expected_grad_input);
        assert_tensors_eq(&grad_weights, &expected_grad_weights);
        assert_tensors_eq(&grad_bias, &expected_grad_bias);
    }
}
