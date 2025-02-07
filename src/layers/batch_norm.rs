use crate::{LayerBackwardContext, LayerForwardContext, Tensor};

/// Implements Batch Normalization layer.
#[derive(Debug)]
pub struct BatchNorm {
    /// Small constant for numerical stability
    pub eps: f32,
    /// Momentum for running statistics updates
    pub momentum: f32,
    /// Number of input features/channels
    pub num_features: usize,
    /// Learnable scale parameter
    pub gamma: Tensor,
    /// Learnable shift parameter
    pub beta: Tensor,
    /// Accumulated gradients for gamma
    pub grad_gamma: Tensor,
    /// Accumulated gradients for beta
    pub grad_beta: Tensor,
    /// Running mean for inference
    pub running_mean: Tensor,
    /// Running variance for inference
    pub running_var: Tensor,
}

#[derive(Debug)]
pub struct BatchNormForwardContext {
    /// Normalized values before scaling and shifting
    pub normalized: Tensor,
    /// Standard deviation for each feature
    pub std: Tensor,
    /// Mean of current batch
    pub batch_mean: Tensor,
    /// Variance of current batch
    pub batch_var: Tensor,
}

#[derive(Debug)]
pub struct BatchNormBackwardContext {
    pub grad_gamma: Tensor,
    pub grad_beta: Tensor,
}

impl BatchNorm {
    /// Creates a new BatchNorm layer with specified number of features
    pub fn new(num_features: usize) -> Self {
        let gamma = Tensor::new_with_shape(vec![1.0; num_features], (num_features, 1));
        let beta = Tensor::new_with_shape(vec![0.0; num_features], (num_features, 1));
        let grad_gamma = Tensor::zeros_like(&gamma);
        let grad_beta = Tensor::zeros_like(&beta);
        let running_mean = Tensor::new_with_shape(vec![0.0; num_features], (num_features, 1));
        let running_var = Tensor::new_with_shape(vec![1.0; num_features], (num_features, 1));

        BatchNorm {
            eps: 1e-6,
            momentum: 0.1,
            num_features,
            gamma,
            beta,
            running_mean,
            running_var,
            grad_gamma,
            grad_beta,
        }
    }

    pub fn forward(&self, input: Tensor, training: bool) -> (Tensor, LayerForwardContext) {
        let input_shape = input.shape;
        let (batch_size, features) = input.shape;
        let spatial_size = features / self.num_features;

        // During inference, use running statistics
        if !training {
            let mut output = input;
            for i in 0..batch_size {
                for f in 0..self.num_features {
                    for s in 0..spatial_size {
                        let idx = i * features + f * spatial_size + s;
                        output.data[idx] = (output.data[idx] - self.running_mean.data[f])
                            / (self.running_var.data[f] + self.eps).sqrt()
                            * self.gamma.data[f]
                            + self.beta.data[f];
                    }
                }
            }
            return (
                output,
                LayerForwardContext::BatchNorm(BatchNormForwardContext {
                    normalized: Tensor::zeros(input_shape),
                    std: Tensor::zeros((self.num_features, 1)),
                    batch_mean: Tensor::zeros((self.num_features, 1)),
                    batch_var: Tensor::zeros((self.num_features, 1)),
                }),
            );
        }

        // Compute mean for each feature
        let mut mean = vec![0.0; self.num_features];
        for i in 0..batch_size {
            for f in 0..self.num_features {
                for s in 0..spatial_size {
                    mean[f] += input.data[i * features + f * spatial_size + s];
                }
            }
        }
        for f in 0..self.num_features {
            mean[f] /= (batch_size * spatial_size) as f32;
        }

        // Compute variance for each feature
        let mut var = vec![0.0; self.num_features];
        let mut x_centered = input;
        for i in 0..batch_size {
            for f in 0..self.num_features {
                for s in 0..spatial_size {
                    let idx = i * features + f * spatial_size + s;
                    x_centered.data[idx] -= mean[f];
                    var[f] += x_centered.data[idx] * x_centered.data[idx];
                }
            }
        }
        for f in 0..self.num_features {
            var[f] /= (batch_size * spatial_size) as f32;
        }

        // Compute standard deviation
        let mut std = var.clone();
        for f in 0..self.num_features {
            std[f] = (var[f] + self.eps).sqrt();
        }

        // Normalize input
        let mut normalized = x_centered;
        for i in 0..batch_size {
            for f in 0..self.num_features {
                for s in 0..spatial_size {
                    let idx = i * features + f * spatial_size + s;
                    normalized.data[idx] /= std[f];
                }
            }
        }

        // Scale and shift normalized values
        let mut output = normalized.clone();
        for i in 0..batch_size {
            for f in 0..self.num_features {
                for s in 0..spatial_size {
                    let idx = i * features + f * spatial_size + s;
                    output.data[idx] = output.data[idx] * self.gamma.data[f] + self.beta.data[f];
                }
            }
        }

        (
            output,
            LayerForwardContext::BatchNorm(BatchNormForwardContext {
                normalized,
                std: Tensor::new_with_shape(std, (self.num_features, 1)),
                batch_mean: Tensor::new_with_shape(mean, (self.num_features, 1)),
                batch_var: Tensor::new_with_shape(var, (self.num_features, 1)),
            }),
        )
    }

    pub fn backward(
        &self,
        grad_output: Tensor,
        context: &BatchNormForwardContext,
    ) -> (Tensor, LayerBackwardContext) {
        let (batch_size, features) = grad_output.shape;
        let spatial_size = features / self.num_features;
        let n = (batch_size * spatial_size) as f32;

        // Compute gradients for gamma and beta
        let mut grad_gamma = vec![0.0; self.num_features];
        let mut grad_beta = vec![0.0; self.num_features];
        for i in 0..batch_size {
            for f in 0..self.num_features {
                for s in 0..spatial_size {
                    let idx = i * features + f * spatial_size + s;
                    grad_gamma[f] += grad_output.data[idx] * context.normalized.data[idx];
                    grad_beta[f] += grad_output.data[idx];
                }
            }
        }

        // Scale gradients by gamma
        let mut grad_normalized = grad_output;
        for i in 0..batch_size {
            for f in 0..self.num_features {
                for s in 0..spatial_size {
                    let idx = i * features + f * spatial_size + s;
                    grad_normalized.data[idx] *= self.gamma.data[f];
                }
            }
        }

        // Compute input gradients
        let mut grad_input = grad_normalized;
        for f in 0..self.num_features {
            let inv_std = 1.0 / context.std.data[f];

            let mut sum_grad = 0.0;
            let mut sum_grad_normalized = 0.0;
            for i in 0..batch_size {
                for s in 0..spatial_size {
                    let idx = i * features + f * spatial_size + s;
                    sum_grad += grad_input.data[idx];
                    sum_grad_normalized += grad_input.data[idx] * context.normalized.data[idx];
                }
            }

            for i in 0..batch_size {
                for s in 0..spatial_size {
                    let idx: usize = i * features + f * spatial_size + s;
                    grad_input.data[idx] = inv_std
                        * (n * grad_input.data[idx]
                            - sum_grad
                            - context.normalized.data[idx] * sum_grad_normalized)
                        / n
                }
            }
        }

        (
            grad_input,
            LayerBackwardContext::BatchNorm(BatchNormBackwardContext {
                grad_gamma: Tensor::new_with_shape(grad_gamma, (self.num_features, 1)),
                grad_beta: Tensor::new_with_shape(grad_beta, (self.num_features, 1)),
            }),
        )
    }

    /// Accumulates gradients from backward pass
    pub fn update_grads(&mut self, context: &BatchNormBackwardContext) {
        for (dest, src) in self
            .grad_gamma
            .data
            .iter_mut()
            .zip(context.grad_gamma.data.iter())
        {
            *dest += src;
        }

        for (dest, src) in self
            .grad_beta
            .data
            .iter_mut()
            .zip(context.grad_beta.data.iter())
        {
            *dest += src;
        }
    }

    /// Returns references to learnable parameters
    pub fn get_parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma, &self.beta]
    }

    /// Returns mutable parameter references paired with their gradients
    pub fn get_parameter_pairs(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        vec![
            (&mut self.gamma, &self.grad_gamma),
            (&mut self.beta, &self.grad_beta),
        ]
    }

    /// Resets accumulated gradients to zero
    pub fn clear_grads(&mut self) {
        for val in self.grad_gamma.data.iter_mut() {
            *val = 0.0;
        }
        for val in self.grad_beta.data.iter_mut() {
            *val = 0.0;
        }
    }

    /// Updates running mean and variance using batch statistics
    pub fn update_running_stats(&mut self, context: &BatchNormForwardContext) {
        for f in 0..self.num_features {
            self.running_mean.data[f] = (1.0 - self.momentum) * self.running_mean.data[f]
                + self.momentum * context.batch_mean.data[f];
            self.running_var.data[f] = (1.0 - self.momentum) * self.running_var.data[f]
                + self.momentum * context.batch_var.data[f];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_close;

    #[test]
    fn test_forward_dense_output() {
        let mut bn = BatchNorm::new(3);
        bn.gamma = Tensor::new_with_shape(vec![0.5, 1.5, -2.5], (3, 1));
        bn.beta = Tensor::new_with_shape(vec![0.5, 1.5, -2.5], (3, 1));

        let input_data = vec![
            1.0, 2.0, 3.0, //
            2.0, 3.0, 4.0, //
            3.0, 4.0, 5.0, //
            4.0, 5.0, 6.0, //
        ];
        let input = Tensor::new_with_shape(input_data, (4, 3));

        let (output, context) = bn.forward(input.clone(), true);
        let bn_context = match context {
            LayerForwardContext::BatchNorm(ctx) => ctx,
            _ => panic!(),
        };

        assert_eq!(output.shape, (4, 3));

        for i in 0..3 {
            let mut feature_mean = 0.0;
            let mut feature_var = 0.0;

            for j in 0..4 {
                feature_mean += input.data[j * 3 + i];
            }
            feature_mean /= 4.0;

            for j in 0..4 {
                feature_var += (input.data[j * 3 + i] - feature_mean).powi(2);
            }
            feature_var /= 4.0;

            assert_close(bn_context.batch_mean.data[i], feature_mean, 1e-5);
            assert_close(bn_context.batch_var.data[i], feature_var, 1e-5);

            assert_close(bn_context.std.data[i], (feature_var + bn.eps).sqrt(), 1e-5);

            let mut norm_mean = 0.0;
            let mut norm_var = 0.0;

            for j in 0..4 {
                norm_mean += bn_context.normalized.data[j * 3 + i];
            }
            norm_mean /= 4.0;

            for j in 0..4 {
                norm_var += (bn_context.normalized.data[j * 3 + i] - norm_mean).powi(2);
            }
            norm_var /= 4.0;

            assert_close(norm_mean, 0.0, 1e-5);
            assert_close(norm_var, 1.0, 1e-5);
        }

        for i in 0..3 {
            let mut mean = 0.0;
            let mut var = 0.0;

            for j in 0..4 {
                mean += output.data[j * 3 + i];
            }
            mean /= 4.0;

            for j in 0..4 {
                var += (output.data[j * 3 + i] - mean).powi(2);
            }
            var /= 4.0;

            assert_close(mean, bn.beta.data[i], 1e-5);
            assert_close(var, bn.gamma.data[i].powi(2), 1e-5);
        }
    }

    #[test]
    fn test_forward_conv_output() {
        let mut bn = BatchNorm::new(2);
        bn.gamma = Tensor::new_with_shape(vec![0.5, 1.5], (2, 1));
        bn.beta = Tensor::new_with_shape(vec![0.5, 1.5], (2, 1));

        // 2 features, 2 spatial size
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, //
            2.0, 3.0, 4.0, 5.0, //
            3.0, 4.0, 5.0, 6.0, //
            4.0, 5.0, 6.0, 7.0, //
        ];
        let input = Tensor::new_with_shape(input_data, (4, 4));

        let (output, context) = bn.forward(input.clone(), true);
        let bn_context = match context {
            LayerForwardContext::BatchNorm(ctx) => ctx,
            _ => panic!(),
        };

        assert_eq!(output.shape, (4, 4));

        for i in 0..2 {
            let mut feature_mean = 0.0;
            let mut feature_var = 0.0;
            let feature_start = i * 2;

            for spatial in 0..2 {
                for batch in 0..4 {
                    feature_mean += input.data[batch * 4 + feature_start + spatial];
                }
            }
            feature_mean /= 8.0; // 2 spatial positions * 4 batch size

            for spatial in 0..2 {
                for batch in 0..4 {
                    feature_var +=
                        (input.data[batch * 4 + feature_start + spatial] - feature_mean).powi(2);
                }
            }
            feature_var /= 8.0;

            assert_close(bn_context.batch_mean.data[i], feature_mean, 1e-5);
            assert_close(bn_context.batch_var.data[i], feature_var, 1e-5);

            assert_close(bn_context.std.data[i], (feature_var + bn.eps).sqrt(), 1e-5);

            let mut norm_mean = 0.0;
            let mut norm_var = 0.0;

            for spatial in 0..2 {
                for batch in 0..4 {
                    norm_mean += bn_context.normalized.data[batch * 4 + feature_start + spatial];
                }
            }
            norm_mean /= 8.0;

            for spatial in 0..2 {
                for batch in 0..4 {
                    norm_var += (bn_context.normalized.data[batch * 4 + feature_start + spatial]
                        - norm_mean)
                        .powi(2);
                }
            }
            norm_var /= 8.0;

            assert_close(norm_mean, 0.0, 1e-5);
            assert_close(norm_var, 1.0, 1e-5);

            let mut output_mean = 0.0;
            let mut output_var = 0.0;

            for spatial in 0..2 {
                for batch in 0..4 {
                    output_mean += output.data[batch * 4 + feature_start + spatial];
                }
            }
            output_mean /= 8.0;

            for spatial in 0..2 {
                for batch in 0..4 {
                    output_var +=
                        (output.data[batch * 4 + feature_start + spatial] - output_mean).powi(2);
                }
            }
            output_var /= 8.0;

            assert_close(output_mean, bn.beta.data[i], 1e-5);
            assert_close(output_var, bn.gamma.data[i].powi(2), 1e-5);
        }
    }

    #[test]
    fn test_backward_dense_output() {
        let mut bn = BatchNorm::new(3);
        bn.gamma = Tensor::new_with_shape(vec![0.5, 1.5, -2.5], (3, 1));
        bn.beta = Tensor::new_with_shape(vec![0.5, 1.5, -2.5], (3, 1));

        let input_data = vec![
            1.0, 2.0, 3.0, //
            2.0, 3.0, 4.0, //
            3.0, 4.0, 5.0, //
            4.0, 5.0, 6.0, //
        ];
        let input = Tensor::new_with_shape(input_data.clone(), (4, 3));

        let grad_output_data = vec![
            100.0, 200.0, -300.0, //
            200.0, -300.0, 400.0, //
            -100.0, 400.0, 100.0, //
            400.0, 100.0, 200.0, //
        ];
        let grad_output = Tensor::new_with_shape(grad_output_data, (4, 3));

        let (_, context) = bn.forward(input.clone(), true);
        let bn_context = match context {
            LayerForwardContext::BatchNorm(ctx) => ctx,
            _ => panic!(),
        };

        let (grad_input, back_context) = bn.backward(grad_output.clone(), &bn_context);
        let bn_back_context = match back_context {
            LayerBackwardContext::BatchNorm(ctx) => ctx,
            _ => panic!(),
        };

        let epsilon: f64 = 1e-3;
        for i in 0..3 {
            // Verify output using differentials
            for j in 0..4 {
                let mut perturbed_data = input_data.clone();
                perturbed_data[j * 3 + i] += epsilon as f32;
                let perturbed_input = Tensor::new_with_shape(perturbed_data, (4, 3));

                let (output1, _) = bn.forward(input.clone(), true);
                let (output2, _) = bn.forward(perturbed_input, true);

                let mut loss1: f64 = 0.0;
                let mut loss2: f64 = 0.0;
                for k in 0..output1.data.len() {
                    loss1 += (output1.data[k] as f64) * (grad_output.data[k] as f64);
                    loss2 += (output2.data[k] as f64) * (grad_output.data[k] as f64);
                }

                // epsilon needs to be quite large here
                let numeric_grad = (loss2 - loss1) / epsilon;
                assert_close(grad_input.data[j * 3 + i], numeric_grad as f32, 2e-1);
            }

            let n = 4.0; // batch size
            let gamma = bn.gamma.data[i];
            let inv_std = 1.0 / bn_context.std.data[i];

            let mut sum_grad = 0.0;
            let mut sum_grad_normalized = 0.0;
            for j in 0..4 {
                sum_grad += grad_output.data[j * 3 + i];
                sum_grad_normalized +=
                    grad_output.data[j * 3 + i] * bn_context.normalized.data[j * 3 + i];
            }

            for j in 0..4 {
                let normalized = bn_context.normalized.data[j * 3 + i];
                let expected_grad = gamma
                    * inv_std
                    * (n * grad_output.data[j * 3 + i]
                        - sum_grad
                        - normalized * sum_grad_normalized)
                    / n;

                assert_close(grad_input.data[j * 3 + i], expected_grad, 1e-4);
            }

            {
                let mut bn_plus = BatchNorm::new(3);
                bn_plus.gamma = bn.gamma.clone();
                let mut beta_plus = bn.beta.clone();
                beta_plus.data[i] += epsilon as f32;
                bn_plus.beta = beta_plus;

                let (output1, _) = bn.forward(input.clone(), true);
                let (output2, _) = bn_plus.forward(input.clone(), true);

                let mut loss1: f64 = 0.0;
                let mut loss2: f64 = 0.0;
                for k in 0..output1.data.len() {
                    loss1 += (output1.data[k] as f64) * (grad_output.data[k] as f64);
                    loss2 += (output2.data[k] as f64) * (grad_output.data[k] as f64);
                }

                let numeric_grad = (loss2 - loss1) / epsilon;
                assert_close(bn_back_context.grad_beta.data[i], numeric_grad as f32, 2e-1);
            }

            {
                let mut bn_plus = BatchNorm::new(3);
                bn_plus.beta = bn.beta.clone();
                let mut gamma_plus = bn.gamma.clone();
                gamma_plus.data[i] += epsilon as f32;
                bn_plus.gamma = gamma_plus;

                let (output1, _) = bn.forward(input.clone(), true);
                let (output2, _) = bn_plus.forward(input.clone(), true);

                let mut loss1: f64 = 0.0;
                let mut loss2: f64 = 0.0;
                for k in 0..output1.data.len() {
                    loss1 += (output1.data[k] as f64) * (grad_output.data[k] as f64);
                    loss2 += (output2.data[k] as f64) * (grad_output.data[k] as f64);
                }

                let numeric_grad = (loss2 - loss1) / epsilon;
                assert_close(
                    bn_back_context.grad_gamma.data[i],
                    numeric_grad as f32,
                    2e-1,
                );
            }

            let mut expected_grad_beta: f32 = 0.0;
            for j in 0..4 {
                expected_grad_beta += grad_output.data[j * 3 + i];
            }
            assert_close(bn_back_context.grad_beta.data[i], expected_grad_beta, 1e-5);

            let mut expected_grad_gamma: f32 = 0.0;
            for j in 0..4 {
                expected_grad_gamma +=
                    grad_output.data[j * 3 + i] * bn_context.normalized.data[j * 3 + i];
            }

            assert_close(
                bn_back_context.grad_gamma.data[i],
                expected_grad_gamma,
                1e-5,
            );
        }
    }

    #[test]
    fn test_backward_conv_output() {
        let mut bn = BatchNorm::new(2);
        bn.gamma = Tensor::new_with_shape(vec![0.5, 1.5], (2, 1));
        bn.beta = Tensor::new_with_shape(vec![0.5, 1.5], (2, 1));

        let input_data = vec![
            1.0, 2.0, //
            2.0, 3.0, //
            3.0, 4.0, //
            4.0, 5.0, //
        ];
        let input = Tensor::new_with_shape(input_data.clone(), (4, 2));

        let grad_output_data = vec![
            100.0, 200.0, //
            200.0, -300.0, //
            -100.0, 400.0, //
            400.0, 100.0, //
        ];
        let grad_output = Tensor::new_with_shape(grad_output_data, (4, 2));

        let (_, context) = bn.forward(input.clone(), true);
        let bn_context = match context {
            LayerForwardContext::BatchNorm(ctx) => ctx,
            _ => panic!(),
        };

        let (grad_input, back_context) = bn.backward(grad_output.clone(), &bn_context);
        let bn_back_context = match back_context {
            LayerBackwardContext::BatchNorm(ctx) => ctx,
            _ => panic!(),
        };

        let epsilon: f64 = 1e-3;
        for i in 0..2 {
            // Verify output using differentials
            for j in 0..4 {
                let mut perturbed_data = input_data.clone();
                perturbed_data[j * 2 + i] += epsilon as f32;
                let perturbed_input = Tensor::new_with_shape(perturbed_data, (4, 2));

                let (output1, _) = bn.forward(input.clone(), true);
                let (output2, _) = bn.forward(perturbed_input, true);

                let mut loss1: f64 = 0.0;
                let mut loss2: f64 = 0.0;
                for k in 0..output1.data.len() {
                    loss1 += (output1.data[k] as f64) * (grad_output.data[k] as f64);
                    loss2 += (output2.data[k] as f64) * (grad_output.data[k] as f64);
                }

                // epsilon needs to be quite large here
                let numeric_grad = (loss2 - loss1) / epsilon;
                assert_close(grad_input.data[j * 2 + i], numeric_grad as f32, 2e-1);
            }

            let n = 4.0;
            let gamma = bn.gamma.data[i];
            let inv_std = 1.0 / bn_context.std.data[i];

            let mut sum_grad = 0.0;
            let mut sum_grad_normalized = 0.0;
            for j in 0..4 {
                sum_grad += grad_output.data[j * 2 + i];
                sum_grad_normalized +=
                    grad_output.data[j * 2 + i] * bn_context.normalized.data[j * 2 + i];
            }

            for j in 0..4 {
                let normalized = bn_context.normalized.data[j * 2 + i];
                let expected_grad = gamma
                    * inv_std
                    * (n * grad_output.data[j * 2 + i]
                        - sum_grad
                        - normalized * sum_grad_normalized)
                    / n;
                assert_close(grad_input.data[j * 2 + i], expected_grad, 1e-4);
            }

            {
                let mut bn_plus = BatchNorm::new(2);
                bn_plus.gamma = bn.gamma.clone();
                let mut beta_plus = bn.beta.clone();
                beta_plus.data[i] += epsilon as f32;
                bn_plus.beta = beta_plus;

                let (output1, _) = bn.forward(input.clone(), true);
                let (output2, _) = bn_plus.forward(input.clone(), true);

                let mut loss1: f64 = 0.0;
                let mut loss2: f64 = 0.0;
                for k in 0..output1.data.len() {
                    loss1 += (output1.data[k] as f64) * (grad_output.data[k] as f64);
                    loss2 += (output2.data[k] as f64) * (grad_output.data[k] as f64);
                }

                let numeric_grad = (loss2 - loss1) / epsilon;
                assert_close(bn_back_context.grad_beta.data[i], numeric_grad as f32, 2e-1);
            }

            {
                let mut bn_plus = BatchNorm::new(2);
                bn_plus.beta = bn.beta.clone();
                let mut gamma_plus = bn.gamma.clone();
                gamma_plus.data[i] += epsilon as f32;
                bn_plus.gamma = gamma_plus;

                let (output1, _) = bn.forward(input.clone(), true);
                let (output2, _) = bn_plus.forward(input.clone(), true);

                let mut loss1: f64 = 0.0;
                let mut loss2: f64 = 0.0;
                for k in 0..output1.data.len() {
                    loss1 += (output1.data[k] as f64) * (grad_output.data[k] as f64);
                    loss2 += (output2.data[k] as f64) * (grad_output.data[k] as f64);
                }

                let numeric_grad = (loss2 - loss1) / epsilon;
                assert_close(
                    bn_back_context.grad_gamma.data[i],
                    numeric_grad as f32,
                    2e-1,
                );
            }

            let mut expected_grad_beta: f32 = 0.0;
            for j in 0..4 {
                expected_grad_beta += grad_output.data[j * 2 + i];
            }
            assert_close(bn_back_context.grad_beta.data[i], expected_grad_beta, 1e-5);

            let mut expected_grad_gamma: f32 = 0.0;
            for j in 0..4 {
                expected_grad_gamma +=
                    grad_output.data[j * 2 + i] * bn_context.normalized.data[j * 2 + i];
            }
            assert_close(
                bn_back_context.grad_gamma.data[i],
                expected_grad_gamma,
                1e-5,
            );
        }
    }
}
