use crate::{
    calculate_loss, calculate_number_of_correct_outputs, create_one_hot, load_batch,
    parse_vector_str, prepare_data_iteration, LayerBackwardContext, LayerForwardContext,
    LearningRateScheduler, Network, NormalizationParams, SchedulerOutput, Tensor,
};
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Instant;

/// Configuration parameters for training
pub struct TrainingConfig {
    /// Number of samples in each minibatch
    pub minibatch_size: usize,
    /// Number of minibatches to process in parallel
    pub number_of_minibatches: usize,
    /// Label smoothing factor for regularization
    pub label_smoothing: f32,
    /// Number of training epochs
    pub epochs: usize,
    /// Learning rate scheduling strategy
    pub scheduler: LearningRateScheduler,
}

impl TrainingConfig {
    /// Gets current learning rate from scheduler
    pub fn get_lr(
        self: &TrainingConfig,
        current_step: usize,
        total_steps: usize,
    ) -> SchedulerOutput {
        self.scheduler.get_lr(current_step, total_steps)
    }
}

/// Results from processing a single batch
pub struct BatchResults {
    /// Network output after forward pass
    pub output: Tensor,
    /// Number of correct predictions
    pub correct: f32,
    /// Loss value
    pub loss: f32,
    /// Layer contexts from forward pass
    pub forward_contexts: Option<Vec<LayerForwardContext>>,
    /// Layer contexts from backward pass
    pub backward_contexts: Option<Vec<LayerBackwardContext>>,
}

/// Processes a single batch through the network
///
/// # Arguments
/// * `network` - Neural network model
/// * `inputs` - Batch input data
/// * `targets` - Target labels
/// * `training` - Whether in training mode
/// * `label_smoothing` - Smoothing factor for labels
pub fn process_batch(
    network: &Network,
    inputs: Vec<f32>,
    targets: Vec<f32>,
    vector_size: usize,
    label_smoothing: f32,
    training: bool,
) -> BatchResults {
    let minibatch_size = targets.len();
    let input = Tensor::new_with_shape(inputs, (minibatch_size, vector_size));

    // Forward pass
    let (mut output, forward_contexts) = network.forward(input, training);
    output = output.softmax();

    let correct = calculate_number_of_correct_outputs(&output, &targets);
    let loss = calculate_loss(&output, &targets);

    // Backward pass if training
    let backward_contexts = if training {
        let one_hot = create_one_hot(&targets, label_smoothing);

        // Compute gradients
        let mut output_grad_vec = Vec::with_capacity(minibatch_size * 10);
        for i in 0..minibatch_size {
            for j in 0..10 {
                output_grad_vec.push(output.data[i * 10 + j] - one_hot[i * 10 + j]);
            }
        }

        let output_grad = Tensor::new_with_shape(output_grad_vec, (minibatch_size, 10));
        let (_, backward_contexts) = network.backward(output_grad, &forward_contexts);

        Some(backward_contexts)
    } else {
        None
    };

    BatchResults {
        output,
        correct,
        loss,
        forward_contexts: Some(forward_contexts),
        backward_contexts,
    }
}

/// Processes multiple batches in parallel
///
/// # Arguments
/// * `network` - Thread-safe network reference
/// * `batches` - Vector of (input, target) pairs
/// * `label_smoothing` - Smoothing factor
/// * `training` - Whether in training mode
pub fn process_batches(
    network: &Arc<RwLock<Network>>,
    batches: &[(Vec<f32>, Vec<f32>)],
    vector_size: usize,
    label_smoothing: f32,
    training: bool,
) -> (
    Vec<Vec<LayerForwardContext>>,
    Vec<Vec<LayerBackwardContext>>,
    f32,
    f32,
) {
    let mut handles = Vec::with_capacity(batches.len());
    let mut total_accuracy = 0.0;
    let mut total_loss = 0.0;

    // Process each batch in a separate thread
    for batch in batches {
        let network_arc = Arc::clone(network);
        let (inputs, targets) = batch.clone();

        let handle = thread::spawn(move || {
            let network_guard = network_arc.read().unwrap();
            let results = process_batch(
                &network_guard,
                inputs,
                targets,
                vector_size,
                label_smoothing,
                training,
            );

            (
                results.forward_contexts.unwrap(),
                results.backward_contexts.unwrap_or(vec![]),
                results.correct,
                results.loss,
            )
        });

        handles.push(handle);
    }

    // Collect results from all threads
    let mut forward_contexts_vec = Vec::with_capacity(batches.len());
    let mut backward_contexts_vec = Vec::with_capacity(batches.len());

    for handle in handles {
        let (forward_contexts, backward_contexts, accuracy, loss) = handle.join().unwrap();
        forward_contexts_vec.push(forward_contexts);
        backward_contexts_vec.push(backward_contexts);
        total_accuracy += accuracy;
        total_loss += loss;
    }

    (
        forward_contexts_vec,
        backward_contexts_vec,
        total_accuracy,
        total_loss,
    )
}

/// Statistics from a training epoch
pub struct TrainingStats {
    /// Classification accuracy
    pub accuracy: f32,
    /// Average loss value
    pub loss: f32,
    /// Time taken for epoch
    pub duration: std::time::Duration,
}

/// Trains network for one epoch
///
/// # Arguments
/// * `network` - Neural network model
/// * `config` - Training configuration
/// * `dataset` - Training data as (labels, vectors)
/// * `norm_params` - Normalization parameters
/// * `current_step` - Current training step
/// * `total_steps` - Total training steps
pub fn train_epoch(
    network: &Arc<RwLock<Network>>,
    config: &TrainingConfig,
    dataset: (&[String], &[String]),
    norm_params: &NormalizationParams,
    current_step: &mut usize,
    total_steps: usize,
) -> TrainingStats {
    let epoch_start = Instant::now();
    let (labels, vectors) = dataset;
    let (shuffled_labels, shuffled_vectors) = prepare_data_iteration(labels, vectors, true);

    let vector_size = parse_vector_str(&vectors[0]).len();

    let mut labels_iter = shuffled_labels.iter();
    let mut vectors_iter = shuffled_vectors.iter();

    let mut total_correct = 0.0;
    let mut total_loss = 0.0;
    let mut total_batches = 0;

    loop {
        // Prepare minibatches
        let mut batches = Vec::with_capacity(config.number_of_minibatches);
        for _ in 0..config.number_of_minibatches {
            if let Some(batch) = load_batch(
                &mut vectors_iter,
                &mut labels_iter,
                vector_size,
                config.minibatch_size,
                norm_params,
            ) {
                batches.push(batch);
            } else {
                break;
            }
        }

        if batches.is_empty() {
            break;
        }

        // Get current learning rate and momentum
        let schedule_output = config.scheduler.get_lr(*current_step, total_steps);
        let learning_rate = match schedule_output {
            SchedulerOutput::Basic { learning_rate } => learning_rate,
            SchedulerOutput::WithMomentum {
                learning_rate,
                momentum,
            } => {
                let mut network_guard = network.write().unwrap();
                network_guard.optimizer.set_momentum(momentum);
                learning_rate
            }
        };

        *current_step += batches.len();

        // Process batches and update parameters
        let (forward_contexts_vec, backward_contexts_vec, batch_correct, batch_loss) =
            process_batches(network, &batches, vector_size, config.label_smoothing, true);

        total_correct += batch_correct;
        total_loss += batch_loss;
        total_batches += batches.len();

        network.write().unwrap().update_parameters(
            forward_contexts_vec,
            backward_contexts_vec,
            learning_rate,
        );
    }

    TrainingStats {
        accuracy: (total_correct / labels.len() as f32) * 100.0,
        loss: total_loss / total_batches as f32,
        duration: epoch_start.elapsed(),
    }
}
