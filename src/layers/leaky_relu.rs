use crate::{LayerBackwardContext, LayerForwardContext, Tensor};

/// Implements Leaky ReLU activation function.
#[derive(Debug)]
pub struct LeakyReLU {
    /// Slope for negative input values, can be 0.0 for normal ReLU
    alpha: f32,
}

#[derive(Debug)]
pub struct LeakyReLUForwardContext {
    pub mask: Vec<bool>,
}

#[derive(Debug)]
pub struct LeakyReLUBackwardContext {}

impl LeakyReLU {
    pub fn new(alpha: f32) -> Self {
        LeakyReLU { alpha }
    }

    pub fn forward(&self, input: Tensor, _training: bool) -> (Tensor, LayerForwardContext) {
        let mut mask = Vec::with_capacity(input.data.len());
        let mut output = input;

        // Apply Leaky ReLU
        for val in output.data.iter_mut() {
            mask.push(*val > 0.0);
            if *val <= 0.0 {
                *val *= self.alpha;
            }
        }

        (
            output,
            LayerForwardContext::LeakyReLU(LeakyReLUForwardContext { mask }),
        )
    }

    pub fn backward(
        &self,
        grad_output: Tensor,
        context: &LeakyReLUForwardContext,
    ) -> (Tensor, LayerBackwardContext) {
        let mut output = grad_output;

        // Gradient is 1 for positive inputs, alpha for negative inputs
        for (grad, &was_positive) in output.data.iter_mut().zip(context.mask.iter()) {
            if !was_positive {
                *grad *= self.alpha;
            }
        }

        (
            output,
            LayerBackwardContext::LeakyReLU(LeakyReLUBackwardContext {}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_tensors_eq;

    #[test]
    fn test_leakyrelu_forward_1() {
        let input = Tensor::new_with_shape(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], (1, 6));

        let leakyrelu = LeakyReLU::new(0.1);
        let (output, _) = leakyrelu.forward(input, false);

        let expected = Tensor::new_with_shape(vec![1.0, -0.2, 3.0, -0.4, 5.0, -0.6], (1, 6));
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_leakyrelu_forward_2() {
        let input = Tensor::new_with_shape(
            vec![
                1.0, -2.0, 3.0, //
                -4.0, 5.0, -6.0, //
            ],
            (2, 3),
        );

        let leakyrelu = LeakyReLU::new(0.2);
        let (output, _) = leakyrelu.forward(input, false);

        let expected = Tensor::new_with_shape(
            vec![
                1.0, -0.4, 3.0, //
                -0.8, 5.0, -1.2, //
            ],
            (2, 3),
        );
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_leakyrelu_forward_3() {
        let input = Tensor::new_with_shape(vec![0.0, -0.0, 1.0, -1.0], (1, 4));

        let leakyrelu = LeakyReLU::new(0.1);
        let (output, _) = leakyrelu.forward(input, false);

        let expected = Tensor::new_with_shape(vec![0.0, -0.0, 1.0, -0.1], (1, 4));
        assert_tensors_eq(&output, &expected);
    }

    #[test]
    fn test_leakyrelu_backward_1() {
        let input = Tensor::new_with_shape(vec![1.0, -2.0, 3.0, -4.0], (1, 4));

        let leakyrelu = LeakyReLU::new(0.1);
        let (_, context) = leakyrelu.forward(input, false);

        let context = match context {
            LayerForwardContext::LeakyReLU(context) => context,
            _ => panic!(),
        };

        let grad_output = Tensor::new_with_shape(vec![2.0, 2.0, 2.0, 2.0], (1, 4));

        let (grad_input, _) = leakyrelu.backward(grad_output, &context);

        let expected_grad = Tensor::new_with_shape(vec![2.0, 0.2, 2.0, 0.2], (1, 4));
        assert_tensors_eq(&grad_input, &expected_grad);
    }

    #[test]
    fn test_leakyrelu_backward_2() {
        let input = Tensor::new_with_shape(
            vec![
                1.0, -2.0, 3.0, //
                -4.0, 5.0, -6.0, //
            ],
            (2, 3),
        );

        let leakyrelu = LeakyReLU::new(0.2);
        let (_, context) = leakyrelu.forward(input, false);

        let context = match context {
            LayerForwardContext::LeakyReLU(context) => context,
            _ => panic!(),
        };

        let grad_output = Tensor::new_with_shape(
            vec![
                1.0, 1.0, 1.0, //
                1.0, 1.0, 1.0, //
            ],
            (2, 3),
        );

        let (grad_input, _) = leakyrelu.backward(grad_output, &context);

        let expected_grad = Tensor::new_with_shape(
            vec![
                1.0, 0.2, 1.0, //
                0.2, 1.0, 0.2, //
            ],
            (2, 3),
        );
        assert_tensors_eq(&grad_input, &expected_grad);
    }

    #[test]
    fn test_leakyrelu_backward_3() {
        let input = Tensor::new_with_shape(vec![0.0, -0.0, 1.0, -1.0], (1, 4));

        let leakyrelu = LeakyReLU::new(0.1);
        let (_, context) = leakyrelu.forward(input, false);

        let context = match context {
            LayerForwardContext::LeakyReLU(context) => context,
            _ => panic!(),
        };

        let grad_output = Tensor::new_with_shape(vec![1.0, 1.0, 1.0, 1.0], (1, 4));

        let (grad_input, _) = leakyrelu.backward(grad_output, &context);

        let expected_grad = Tensor::new_with_shape(vec![0.1, 0.1, 1.0, 0.1], (1, 4));
        assert_tensors_eq(&grad_input, &expected_grad);
    }
}
