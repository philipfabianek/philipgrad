use crate::{Layer, LayerBackwardContext, LayerForwardContext, Optimizer, Tensor};
use std::sync::{Arc, RwLock};

/// Neural network implementation supporting arbitrary layer configurations,
/// handles forward/backward passes and parameter updates with thread-safe layer access
pub struct Network {
    /// Thread-safe storage of network layers
    pub layers: Vec<Arc<RwLock<Layer>>>,
    /// Optimizer for parameter updates
    pub optimizer: Box<dyn Optimizer>,
}

impl Network {
    /// Creates a new network with specified optimizer
    pub fn new(optimizer: impl Optimizer + 'static) -> Self {
        Network {
            layers: Vec::new(),
            optimizer: Box::new(optimizer),
        }
    }

    /// Adds a new layer to the network, wrapping it in thread-safe containers
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(Arc::new(RwLock::new(layer)));
    }

    /// Initializes optimizer with all trainable parameters from the network
    pub fn initialize_optimizer(&mut self) {
        let guards: Vec<_> = self
            .layers
            .iter()
            .map(|layer| layer.read().unwrap())
            .collect();

        let mut all_params = Vec::new();
        for guard in &guards {
            all_params.extend(guard.get_parameters());
        }

        self.optimizer.init(&all_params);
    }

    /// Performs forward pass through the network
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `training` - Whether in training mode (affects certain layers)
    ///
    /// # Returns
    /// Tuple of (output tensor, forward contexts needed for backward pass)
    pub fn forward(&self, input: Tensor, training: bool) -> (Tensor, Vec<LayerForwardContext>) {
        let mut current = input;
        let mut contexts = Vec::with_capacity(self.layers.len());

        // Process through each layer sequentially
        for layer in &self.layers {
            let (output, context) = {
                let layer_guard = layer.read().unwrap();
                layer_guard.forward(current, training)
            };
            current = output;

            contexts.push(context);
        }

        (current, contexts)
    }

    /// Performs backward pass through the network
    ///
    /// # Arguments
    /// * `grad_output` - Gradient tensor from loss function
    /// * `forward_contexts` - Contexts saved during forward pass
    ///
    /// # Returns
    /// Tuple of (input gradients, backward contexts for parameter updates)
    pub fn backward(
        &self,
        grad_output: Tensor,
        forward_contexts: &Vec<LayerForwardContext>,
    ) -> (Tensor, Vec<LayerBackwardContext>) {
        let mut current = grad_output;
        let mut contexts = Vec::with_capacity(self.layers.len());

        // Process through layers in reverse order
        for (layer, forward_context) in self.layers.iter().zip(forward_contexts.iter()).rev() {
            let (output, context) = {
                let layer_guard = layer.read().unwrap();
                layer_guard.backward(current, forward_context)
            };
            current = output;

            contexts.push(context);
        }

        (current, contexts)
    }

    /// Updates network parameters using accumulated gradients
    ///
    /// # Arguments
    /// * `forward_contexts_vec` - Forward contexts from multiple batches
    /// * `backward_contexts_vec` - Backward contexts from multiple batches
    /// * `learning_rate` - Current learning rate
    pub fn update_parameters(
        &mut self,
        forward_contexts_vec: Vec<Vec<LayerForwardContext>>,
        backward_contexts_vec: Vec<Vec<LayerBackwardContext>>,
        learning_rate: f32,
    ) {
        for layer_idx in 0..self.layers.len() {
            for (forward_contexts, backward_contexts) in forward_contexts_vec
                .iter()
                .zip(backward_contexts_vec.iter())
            {
                let mut layer = self.layers[layer_idx].write().unwrap();

                let forward_context = &forward_contexts[layer_idx];
                let backward_context = &backward_contexts[backward_contexts.len() - 1 - layer_idx];

                layer.update_state(forward_context, backward_context);
            }
        }

        // Collect all parameter-gradient pairs
        let mut guards: Vec<_> = self
            .layers
            .iter()
            .map(|layer| layer.write().unwrap())
            .collect();
        let mut param_pairs = Vec::new();
        for guard in &mut guards {
            param_pairs.extend(guard.get_parameter_pairs().into_iter());
        }

        // Update parameters using optimizer
        self.optimizer
            .update_parameters(&mut param_pairs, learning_rate);

        // Clear gradients after update
        for guard in &mut guards {
            guard.clear_grads();
        }
    }
}
