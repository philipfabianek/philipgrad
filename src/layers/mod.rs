use crate::Tensor;

mod batch_norm;
mod conv;
mod dense;
mod dropout;
mod leaky_relu;
mod max_pool;

pub use batch_norm::{BatchNorm, BatchNormBackwardContext, BatchNormForwardContext};
pub use conv::{Conv, ConvBackwardContext, ConvForwardContext};
pub use dense::{Dense, DenseBackwardContext, DenseForwardContext};
pub use dropout::{Dropout, DropoutBackwardContext, DropoutForwardContext};
pub use leaky_relu::{LeakyReLU, LeakyReLUBackwardContext, LeakyReLUForwardContext};
pub use max_pool::{MaxPool, MaxPoolBackwardContext, MaxPoolForwardContext};

/// Available neural network layers
#[derive(Debug)]
pub enum Layer {
    Conv(Conv),
    Dense(Dense),
    Dropout(Dropout),
    MaxPool(MaxPool),
    LeakyReLU(LeakyReLU),
    BatchNorm(BatchNorm),
}

/// Layer-specific context from forward pass for backpropagation
#[derive(Debug)]
pub enum LayerForwardContext {
    Conv(ConvForwardContext),
    Dense(DenseForwardContext),
    Dropout(DropoutForwardContext),
    MaxPool(MaxPoolForwardContext),
    LeakyReLU(LeakyReLUForwardContext),
    BatchNorm(BatchNormForwardContext),
}

/// Layer-specific gradients and other backward pass information
#[derive(Debug)]
pub enum LayerBackwardContext {
    Conv(ConvBackwardContext),
    Dense(DenseBackwardContext),
    Dropout(DropoutBackwardContext),
    MaxPool(MaxPoolBackwardContext),
    LeakyReLU(LeakyReLUBackwardContext),
    BatchNorm(BatchNormBackwardContext),
}

impl Layer {
    /// Performs forward pass through the layer
    /// Returns output tensor and context needed for backward pass
    pub fn forward(&self, input: Tensor, training: bool) -> (Tensor, LayerForwardContext) {
        match self {
            Layer::BatchNorm(batchnorm) => batchnorm.forward(input, training),
            Layer::Conv(conv) => conv.forward(input, training),
            Layer::Dense(dense) => dense.forward(input, training),
            Layer::Dropout(dropout) => dropout.forward(input, training),
            Layer::MaxPool(maxpool) => maxpool.forward(input, training),
            Layer::LeakyReLU(relu) => relu.forward(input, training),
        }
    }

    /// Performs backward pass through the layer
    /// Takes gradient from next layer and forward context, returns input gradient and backward context
    pub fn backward(
        &self,
        grad: Tensor,
        context: &LayerForwardContext,
    ) -> (Tensor, LayerBackwardContext) {
        match (self, context) {
            (Layer::BatchNorm(batchnorm), LayerForwardContext::BatchNorm(context)) => {
                batchnorm.backward(grad, context)
            }
            (Layer::Conv(conv), LayerForwardContext::Conv(context)) => conv.backward(grad, context),
            (Layer::Dense(dense), LayerForwardContext::Dense(context)) => {
                dense.backward(grad, context)
            }
            (Layer::Dropout(dropout), LayerForwardContext::Dropout(context)) => {
                dropout.backward(grad, context)
            }
            (Layer::MaxPool(maxpool), LayerForwardContext::MaxPool(context)) => {
                maxpool.backward(grad, context)
            }
            (Layer::LeakyReLU(relu), LayerForwardContext::LeakyReLU(context)) => {
                relu.backward(grad, context)
            }
            _ => panic!(),
        }
    }

    /// Updates layer state after backward pass
    pub fn update_state(
        &mut self,
        forward_context: &LayerForwardContext,
        backward_context: &LayerBackwardContext,
    ) {
        match (self, forward_context, backward_context) {
            (
                Layer::BatchNorm(batchnorm),
                LayerForwardContext::BatchNorm(forward_context),
                LayerBackwardContext::BatchNorm(backward_context),
            ) => {
                batchnorm.update_running_stats(forward_context);
                batchnorm.update_grads(backward_context);
            }
            (
                Layer::Conv(conv),
                LayerForwardContext::Conv(_forward_context),
                LayerBackwardContext::Conv(backward_context),
            ) => {
                conv.update_grads(backward_context);
            }
            (
                Layer::Dense(dense),
                LayerForwardContext::Dense(_forward_context),
                LayerBackwardContext::Dense(backward_context),
            ) => {
                dense.update_grads(backward_context);
            }
            _ => {}
        };
    }

    /// Returns references to layer's trainable parameters,
    /// used during optimizer initialization
    pub fn get_parameters(&self) -> Vec<&Tensor> {
        match self {
            Layer::BatchNorm(batchnorm) => batchnorm.get_parameters(),
            Layer::Conv(conv) => conv.get_parameters(),
            Layer::Dense(dense) => dense.get_parameters(),
            _ => vec![],
        }
    }

    /// Returns mutable references to parameters paired with their gradients,
    /// used by optimizer to update parameters
    pub fn get_parameter_pairs(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        match self {
            Layer::BatchNorm(batchnorm) => batchnorm.get_parameter_pairs(),
            Layer::Conv(conv) => conv.get_parameter_pairs(),
            Layer::Dense(dense) => dense.get_parameter_pairs(),
            _ => vec![],
        }
    }

    /// Resets accumulated gradients to zero after parameter update
    pub fn clear_grads(&mut self) {
        match self {
            Layer::BatchNorm(batchnorm) => batchnorm.clear_grads(),
            Layer::Conv(conv) => conv.clear_grads(),
            Layer::Dense(dense) => dense.clear_grads(),
            _ => {}
        }
    }
}
