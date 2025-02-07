use crate::{load_batch, load_data_file};
use crate::{
    parse_vector_str, process_batch, process_batches, DatasetPaths, Network, NormalizationParams,
    Tensor, TrainingConfig,
};
use std::error;
use std::fs::File;
use std::io::{Error, Write};
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::thread;

/// Calculates number of correct predictions from model outputs
///
/// # Arguments
/// * `output` - Model output probabilities
/// * `targets` - Ground truth labels
pub fn calculate_number_of_correct_outputs(output: &Tensor, targets: &[f32]) -> f32 {
    let mut correct = 0;
    for i in 0..output.shape.0 {
        let mut max_idx = 0;
        let mut max_val = output.data[i * 10];
        for j in 0..10 {
            if output.data[i * 10 + j] > max_val {
                max_val = output.data[i * 10 + j];
                max_idx = j;
            }
        }
        if max_idx as f32 == targets[i] {
            correct += 1;
        }
    }
    correct as f32
}

/// Calculates cross-entropy loss with numerical stability
///
/// # Arguments
/// * `output` - Model output probabilities
/// * `targets` - Ground truth labels
pub fn calculate_loss(output: &Tensor, targets: &[f32]) -> f32 {
    let minibatch_size = output.shape.0;
    let epsilon = 1e-15; // For numerical stability
    let mut loss = 0.0;

    for i in 0..minibatch_size {
        let target_idx = targets[i] as usize;
        let pred = output.data[i * 10 + target_idx].clamp(epsilon, 1.0 - epsilon);
        loss -= pred.ln();
    }
    loss / minibatch_size as f32
}

/// Evaluates model accuracy on a given dataset
///
/// # Arguments
/// * `network` - Neural network model wrapped in Arc<RwLock>
/// * `config` - Training configuration parameters
/// * `dataset` - Tuple of (labels, vectors) as string slices
/// * `norm_params` - Data normalization parameters
///
/// # Returns
/// Accuracy percentage on the dataset
pub fn evaluate_dataset(
    network: &Arc<RwLock<Network>>,
    config: &TrainingConfig,
    dataset: (&[String], &[String]),
    norm_params: &NormalizationParams,
) -> f32 {
    let (labels, vectors) = dataset;

    let vector_size = parse_vector_str(&vectors[0]).len();

    let mut vectors_iter = vectors.iter();
    let mut labels_iter = labels.iter();

    let mut total_correct = 0.0;

    loop {
        // Process data in mini-batches
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

        let results = process_batches(
            network,
            &batches,
            vector_size,
            config.label_smoothing,
            false,
        );

        total_correct += results.2;
    }

    total_correct / labels.len() as f32 * 100.0
}

/// Processes batches of data to generate predictions using multiple threads
///
/// # Arguments
/// * `network` - Neural network model wrapped in Arc<RwLock>
/// * `batches` - Vector of (inputs, targets) tuples
/// * `label_smoothing` - Label smoothing factor
///
/// # Returns
/// Vector of predicted class indices for each batch
fn process_prediction_batches(
    network: &Arc<RwLock<Network>>,
    batches: &[(Vec<f32>, Vec<f32>)],
    vector_size: usize,
    label_smoothing: f32,
) -> Vec<Vec<usize>> {
    let mut handles = Vec::with_capacity(batches.len());

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
                false,
            );
            let output = results.output;

            // Find predicted class (maximum output) for each sample
            let mut batch_predictions = Vec::with_capacity(output.shape.0);
            for i in 0..output.shape.0 {
                let mut max_idx = 0;
                let mut max_val = output.data[i * 10];
                for j in 0..10 {
                    if output.data[i * 10 + j] > max_val {
                        max_val = output.data[i * 10 + j];
                        max_idx = j;
                    }
                }
                batch_predictions.push(max_idx);
            }
            batch_predictions
        });

        handles.push(handle);
    }

    handles.into_iter().map(|h| h.join().unwrap()).collect()
}

/// Generates predictions for all samples in a dataset
///
/// # Arguments
/// * `network` - Neural network model
/// * `config` - Training configuration
/// * `dataset` - Input dataset as (labels, vectors)
/// * `norm_params` - Normalization parameters
///
/// # Returns
/// Vector of predicted class indices for all samples
fn get_class_predictions(
    network: &Arc<RwLock<Network>>,
    config: &TrainingConfig,
    dataset: (&[String], &[String]),
    norm_params: &NormalizationParams,
) -> Vec<usize> {
    let (labels, vectors) = dataset;

    let vector_size = parse_vector_str(&vectors[0]).len();

    let mut vectors_iter = vectors.iter();
    let mut labels_iter = labels.iter();

    let mut predictions = Vec::with_capacity(vectors.len());

    loop {
        // Process data in mini-batches
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

        let batch_predictions =
            process_prediction_batches(network, &batches, vector_size, config.label_smoothing);
        for batch_preds in batch_predictions {
            predictions.extend(batch_preds);
        }
    }

    predictions
}

/// Saves predictions to a CSV file
///
/// # Arguments
/// * `predictions` - Vector of class predictions
/// * `filename` - Output file name
fn save_predictions_to_csv(predictions: &[usize], filename: &str) -> Result<(), Error> {
    let path = Path::new(filename);
    let mut file = File::create(path)?;

    for &pred in predictions {
        writeln!(file, "{}", pred)?;
    }

    Ok(())
}

/// Generates and saves predictions for both training and test datasets
///
/// # Arguments
/// * `network` - Neural network model
/// * `config` - Training configuration
/// * `dataset_paths` - Paths to dataset files
/// * `norm_params` - Normalization parameters
///
/// # Returns
/// Result indicating success or containing error
pub fn save_all_predictions(
    network: &Arc<RwLock<Network>>,
    config: &TrainingConfig,
    dataset_paths: &DatasetPaths,
    norm_params: &NormalizationParams,
) -> Result<(), Box<dyn error::Error>> {
    // Process training data
    let train_labels = load_data_file(&dataset_paths.train_labels);
    let train_vectors = load_data_file(&dataset_paths.train_vectors);

    let train_predictions = get_class_predictions(
        network,
        config,
        (&train_labels, &train_vectors),
        norm_params,
    );
    save_predictions_to_csv(&train_predictions, "train_predictions.csv")?;

    // Process test data
    let test_labels = load_data_file(&dataset_paths.test_labels);
    let test_vectors = load_data_file(&dataset_paths.test_vectors);

    let test_predictions =
        get_class_predictions(network, config, (&test_labels, &test_vectors), norm_params);
    save_predictions_to_csv(&test_predictions, "test_predictions.csv")?;

    Ok(())
}
