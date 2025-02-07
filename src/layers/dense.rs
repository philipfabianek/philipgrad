use crate::{LayerBackwardContext, LayerForwardContext, Tensor};
use rand::distributions::Distribution;
use rand::thread_rng;
use rand_distr::Normal;

/// Implements a fully-connected (dense) neural network layer.
#[derive(Debug)]
pub struct Dense {
    /// Weight matrix for linear transformation
    pub weights: Tensor,
    /// Bias vector added to weighted inputs
    pub bias: Tensor,
    /// Accumulated gradients for weights
    pub grad_weights: Tensor,
    /// Accumulated gradients for bias
    pub grad_bias: Tensor,
}

#[derive(Debug)]
pub struct DenseForwardContext {
    pub input: Tensor,
}

#[derive(Debug)]
pub struct DenseBackwardContext {
    pub grad_weights: Tensor,
    pub grad_bias: Tensor,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self::new_with_alpha(input_size, output_size, 0.0)
    }

    /// Creates a new dense layer with He initialization scaled for LeakyReLU
    pub fn new_with_alpha(input_size: usize, output_size: usize, leaky_relu_alpha: f32) -> Self {
        let mut rng = thread_rng();

        // He initialization adapted for LeakyReLU activation
        let weight_scale = (2.0 / ((1.0 + leaky_relu_alpha.powi(2)) * input_size as f32)).sqrt();
        let normal = Normal::new(0.0, weight_scale).unwrap();

        let weights = (0..input_size * output_size)
            .map(|_| normal.sample(&mut rng))
            .collect();
        let weights = Tensor::new_with_shape(weights, (input_size, output_size));

        let bias = Tensor::new_with_shape(vec![0.0; output_size], (1, output_size));

        let grad_weights = Tensor::zeros_like(&weights);
        let grad_bias = Tensor::zeros_like(&bias);

        Dense {
            weights,
            bias,
            grad_weights,
            grad_bias,
        }
    }

    pub fn forward(&self, input: Tensor, _training: bool) -> (Tensor, LayerForwardContext) {
        let mut output = input.matmul(&self.weights);
        let (rows, columns) = output.shape;

        // Add bias to each output
        for col in 0..columns {
            let bias_val = self.bias.data[col];
            for row in 0..rows {
                output.data[row * columns + col] += bias_val;
            }
        }

        (
            output,
            LayerForwardContext::Dense(DenseForwardContext { input }),
        )
    }

    pub fn backward(
        &self,
        grad_output: Tensor,
        context: &DenseForwardContext,
    ) -> (Tensor, LayerBackwardContext) {
        let rows = grad_output.shape.0;
        let columns = grad_output.shape.1;

        // Compute gradients for input, weights, and bias
        let grad_input = grad_output.matmul(&self.weights.transpose());
        let grad_weights = context.input.transpose().matmul(&grad_output);

        // Sum gradients for bias across batch dimension
        let mut grad_bias_vec = Vec::with_capacity(columns);
        for i in 0..columns {
            let mut sum = 0.0;
            for j in 0..rows {
                sum += grad_output.data[j * columns + i];
            }
            grad_bias_vec.push(sum);
        }

        let grad_bias = Tensor::new_with_shape(grad_bias_vec, (1, columns));

        (
            grad_input,
            LayerBackwardContext::Dense(DenseBackwardContext {
                grad_weights,
                grad_bias,
            }),
        )
    }

    /// Accumulates gradients from backward pass
    pub fn update_grads(&mut self, context: &DenseBackwardContext) {
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

    /// Returns references to learnable parameters
    pub fn get_parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights, &self.bias]
    }

    /// Returns mutable parameter references paired with their gradients
    pub fn get_parameter_pairs(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        vec![
            (&mut self.weights, &self.grad_weights),
            (&mut self.bias, &self.grad_bias),
        ]
    }

    /// Resets accumulated gradients to zero
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
    fn test_dense_forward_1() {
        let input = Tensor::new_with_shape(vec![1.0, 2.0], (1, 2));

        let mut dense = Dense::new(2, 3);
        dense.weights = Tensor::new_with_shape(
            vec![
                1.0, 2.0, 3.0, //
                0.5, -1.0, 1.5, //
            ],
            (2, 3),
        );
        dense.bias = Tensor::new_with_shape(vec![0.1, 0.2, 0.3], (1, 3));

        let (output, _) = dense.forward(input, false);

        assert_eq!(output.shape, (1, 3));

        let expected = Tensor::new_with_shape(vec![2.1, 0.2, 6.3], (1, 3));
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_dense_forward_2() {
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
            ],
            (2, 2),
        );

        let mut dense = Dense::new(2, 2);
        dense.weights = Tensor::new_with_shape(
            vec![
                1.0, 2.0, //
                -1.0, 1.0, //
            ],
            (2, 2),
        );
        dense.bias = Tensor::new_with_shape(vec![0.1, 0.2], (1, 2));

        let (output, _) = dense.forward(input, false);

        assert_eq!(output.shape, (2, 2));

        let expected = Tensor::new_with_shape(
            vec![
                -0.9, 4.2, //
                -0.9, 10.2, //
            ],
            (2, 2),
        );
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_dense_backward_1() {
        let input = Tensor::new_with_shape(vec![1.0, 2.0], (1, 2));

        let mut dense = Dense::new(2, 3);
        dense.weights = Tensor::new_with_shape(
            vec![
                1.0, 2.0, 3.0, //
                0.5, -1.0, 1.5, //
            ],
            (2, 3),
        );

        let (_, context) = dense.forward(input.clone(), false);
        let context = match context {
            LayerForwardContext::Dense(context) => context,
            _ => panic!(),
        };

        let grad_output = Tensor::new_with_shape(vec![1.0, -1.0, 0.5], (1, 3));

        let (grad_input, backward_context) = dense.backward(grad_output, &context);
        let DenseBackwardContext {
            grad_weights,
            grad_bias,
        } = match backward_context {
            LayerBackwardContext::Dense(ctx) => ctx,
            _ => panic!(),
        };

        assert_eq!(grad_input.shape, (1, 2));
        assert_eq!(grad_weights.shape, (2, 3));
        assert_eq!(grad_bias.shape, (1, 3));

        let expected_grad_input = Tensor::new_with_shape(vec![0.5, 2.25], (1, 2));

        let expected_grad_weights = Tensor::new_with_shape(
            vec![
                1.0, -1.0, 0.5, //
                2.0, -2.0, 1.0, //
            ],
            (2, 3),
        );

        let expected_grad_bias = Tensor::new_with_shape(vec![1.0, -1.0, 0.5], (1, 3));

        assert_tensors_eq(&grad_input, &expected_grad_input);
        assert_tensors_eq(&grad_weights, &expected_grad_weights);
        assert_tensors_eq(&grad_bias, &expected_grad_bias);
    }

    #[test]
    fn test_dense_backward_2() {
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
            ],
            (2, 2),
        );

        let mut dense = Dense::new(2, 2);
        dense.weights = Tensor::new_with_shape(
            vec![
                1.0, 2.0, //
                -1.0, 1.0, //
            ],
            (2, 2),
        );

        let (_, context) = dense.forward(input.clone(), false);
        let context = match context {
            LayerForwardContext::Dense(context) => context,
            _ => panic!(),
        };

        let grad_output = Tensor::new_with_shape(
            vec![
                1.0, -1.0, //
                0.5, 0.5, //
            ],
            (2, 2),
        );

        let (grad_input, backward_context) = dense.backward(grad_output, &context);
        let DenseBackwardContext {
            grad_weights,
            grad_bias,
        } = match backward_context {
            LayerBackwardContext::Dense(ctx) => ctx,
            _ => panic!(),
        };

        let expected_grad_input = Tensor::new_with_shape(
            vec![
                -1.0, -2.0, //
                1.5, 0.0, //
            ],
            (2, 2),
        );

        let expected_grad_weights = Tensor::new_with_shape(
            vec![
                2.5, 0.5, //
                4.0, 0.0, //
            ],
            (2, 2),
        );

        let expected_grad_bias = Tensor::new_with_shape(vec![1.5, -0.5], (1, 2));

        assert_tensors_eq(&grad_input, &expected_grad_input);
        assert_tensors_eq(&grad_weights, &expected_grad_weights);
        assert_tensors_eq(&grad_bias, &expected_grad_bias);
    }
}
