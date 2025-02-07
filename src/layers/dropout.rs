use crate::{LayerBackwardContext, LayerForwardContext, Tensor};

use rand::Rng;

/// Implements dropout regularization for neural networks.
/// Randomly deactivates a fraction of neurons during training.
#[derive(Debug)]
pub struct Dropout {
    /// Probability of deactivating each neuron
    pub rate: f32,
    /// Scaling factor applied to active neurons to maintain expected sum
    scale: f32,
}

#[derive(Debug)]
pub struct DropoutForwardContext {
    pub mask: Vec<bool>,
}

#[derive(Debug)]
pub struct DropoutBackwardContext {}

impl Dropout {
    pub fn new(rate: f32) -> Self {
        Dropout {
            rate,
            scale: 1.0 / (1.0 - rate),
        }
    }

    pub fn forward(&self, input: Tensor, training: bool) -> (Tensor, LayerForwardContext) {
        let n = input.data.len();

        // During evaluation, return input unchanged
        if !training {
            return (
                input,
                LayerForwardContext::Dropout(DropoutForwardContext {
                    mask: vec![true; n],
                }),
            );
        }

        let mut output = input;
        let mut rng = rand::thread_rng();
        let mut mask = Vec::with_capacity(n);

        // Generate dropout mask and apply scaling
        for val in output.data.iter_mut() {
            let keep = rng.gen::<f32>() > self.rate;
            mask.push(keep);
            if keep {
                *val *= self.scale;
            } else {
                *val = 0.0;
            }
        }

        (
            output,
            LayerForwardContext::Dropout(DropoutForwardContext { mask }),
        )
    }

    pub fn backward(
        &self,
        grad_output: Tensor,
        context: &DropoutForwardContext,
    ) -> (Tensor, LayerBackwardContext) {
        let mut output = grad_output;

        for (grad, &keep) in output.data.iter_mut().zip(context.mask.iter()) {
            if keep {
                *grad *= self.scale;
            } else {
                *grad = 0.0;
            }
        }

        (
            output,
            LayerBackwardContext::Dropout(DropoutBackwardContext {}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_tensors_eq;

    #[test]
    fn test_dropout_forward_eval_mode() {
        let input = Tensor::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0], (1, 5));
        let dropout = Dropout::new(0.5);

        let (output, context) = dropout.forward(input.clone(), false);

        assert_tensors_eq(&output, &input);

        match context {
            LayerForwardContext::Dropout(ctx) => {
                assert!(ctx.mask.iter().all(|&x| x));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_dropout_forward_training_mode() {
        let size = 100000;
        let input = Tensor::new_with_shape(vec![2.0; size], (1, size));
        let dropout = Dropout::new(0.3);

        let (output, context) = dropout.forward(input.clone(), true);

        match context {
            LayerForwardContext::Dropout(ctx) => {
                let kept_count = ctx.mask.iter().filter(|&&x| x).count();
                let keep_rate = kept_count as f32 / size as f32;

                assert!((keep_rate - 0.7).abs() < 0.1);

                for (val, &keep) in output.data.iter().zip(ctx.mask.iter()) {
                    if keep {
                        assert!((val - 2.0 * (1.0 / 0.7)).abs() < 1e-6,);
                    } else {
                        assert!(*val == 0.0);
                    }
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_dropout_backward() {
        let input = Tensor::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], (1, 4));
        let dropout = Dropout::new(0.5);

        let (_, context) = dropout.forward(input, true);
        let context = match context {
            LayerForwardContext::Dropout(ctx) => ctx,
            _ => panic!(),
        };

        let grad_output = Tensor::new_with_shape(vec![2.0, 2.0, 2.0, 2.0], (1, 4));

        let (grad_input, _) = dropout.backward(grad_output, &context);

        for (grad, &keep) in grad_input.data.iter().zip(context.mask.iter()) {
            if keep {
                assert!((*grad - 2.0 * (1.0 / 0.5)).abs() < 1e-6);
            } else {
                assert!(*grad == 0.0);
            }
        }
    }

    #[test]
    fn test_dropout_consistency() {
        let input = Tensor::new_with_shape(vec![1.0; 10], (1, 10));
        let dropout = Dropout::new(0.5);

        let (output, context) = dropout.forward(input, true);
        let context = match context {
            LayerForwardContext::Dropout(ctx) => ctx,
            _ => panic!(),
        };

        for (val, &keep) in output.data.iter().zip(context.mask.iter()) {
            assert_eq!(*val == 0.0, !keep);
        }
    }
}
