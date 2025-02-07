use philipgrad::{
    evaluate_dataset, layers, load_data_file, parse_arguments, parse_vector_str,
    save_all_predictions, split_data, train_epoch, Adam, Layer, LearningRateScheduler, Network,
    NormalizationParams, TrainingConfig,
};
use std::sync::{Arc, RwLock};
use std::time::Instant;

fn create_network(vector_size: usize) -> Network {
    let height = (vector_size as f32).sqrt();
    let optimizer = Adam::new(0.9, 0.999, 1e-8);
    let mut network = Network::new(optimizer);

    network.add_layer(Layer::Conv(layers::Conv::new_with_alpha(
        1,
        8,
        (3, 3),
        (1, 1),
        (1, 1),
        0.01,
    )));
    network.add_layer(Layer::BatchNorm(layers::BatchNorm::new(8)));
    network.add_layer(Layer::LeakyReLU(layers::LeakyReLU::new(0.01)));

    network.add_layer(Layer::Conv(layers::Conv::new_with_alpha(
        8,
        16,
        (3, 3),
        (1, 1),
        (1, 1),
        0.01,
    )));
    network.add_layer(Layer::BatchNorm(layers::BatchNorm::new(16)));
    network.add_layer(Layer::LeakyReLU(layers::LeakyReLU::new(0.01)));

    network.add_layer(Layer::MaxPool(layers::MaxPool::new(16, 2, 2)));
    network.add_layer(Layer::Dropout(layers::Dropout::new(0.10)));

    let input_size = ((height / 2.0) as usize) * ((height / 2.0) as usize) * 16;
    network.add_layer(Layer::Dense(layers::Dense::new_with_alpha(
        input_size, 512, 0.01,
    )));
    network.add_layer(Layer::BatchNorm(layers::BatchNorm::new(512)));
    network.add_layer(Layer::LeakyReLU(layers::LeakyReLU::new(0.01)));
    network.add_layer(Layer::Dropout(layers::Dropout::new(0.10)));

    network.add_layer(Layer::Dense(layers::Dense::new(512, 10)));

    network.initialize_optimizer();

    network
}

fn main() {
    let start = Instant::now();

    let dataset_paths = match parse_arguments() {
        Ok(paths) => paths,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    let labels = load_data_file(&dataset_paths.train_labels);
    let vectors = load_data_file(&dataset_paths.train_vectors);
    let vector_size = parse_vector_str(&vectors[0]).len();

    let config = TrainingConfig {
        minibatch_size: 64,
        number_of_minibatches: 16,
        label_smoothing: 0.1,
        epochs: 20,
        scheduler: LearningRateScheduler::OneCycle {
            max_lr: 0.03,
            div_factor: 10.0,
            final_div_factor: 100.0,
            pct_start: 0.20,
            base_momentum: 0.85,
            max_momentum: 0.95,
        },
    };

    let network_struct = create_network(vector_size);
    let network = Arc::new(RwLock::new(network_struct));

    println!("\nTraining network, number of epochs: {}", config.epochs);

    let norm_params = NormalizationParams::from_data(&vectors);

    let ((train_labels, train_vectors), (_validation_labels, _validation_vectors)) =
        split_data(&labels, &vectors, 0.0);

    let total_batches = (train_labels.len() + config.minibatch_size - 1) / config.minibatch_size;
    let total_steps = total_batches * config.epochs;
    let mut current_step = 0;

    for epoch in 1..config.epochs + 1 {
        let training_stats = train_epoch(
            &network,
            &config,
            (&train_labels, &train_vectors),
            &norm_params,
            &mut current_step,
            total_steps,
        );

        println!("\n=== Epoch {} Summary ===", epoch);
        println!("Duration: {:.2?}", training_stats.duration);
        println!("Training loss: {:.6}", training_stats.loss);
        println!("Training accuracy: {:.2}%", training_stats.accuracy);

        // let validation_accuracy = evaluate_dataset(
        //     &network,
        //     &config,
        //     (&validation_labels, &validation_vectors),
        //     &norm_params,
        // );

        // println!("Validation accuracy: {:.2}%", validation_accuracy);
    }

    let test_labels = load_data_file(&dataset_paths.test_labels);
    let test_vectors = load_data_file(&dataset_paths.test_vectors);

    let test_accuracy = evaluate_dataset(
        &network,
        &config,
        (&test_labels, &test_vectors),
        &norm_params,
    );

    println!("\n=== Evaluating test dataset ===");
    println!("Test accuracy: {:.2}%", test_accuracy);

    println!("\nSaving predictions...");
    if let Ok(()) = save_all_predictions(&network, &config, &dataset_paths, &norm_params) {
        println!("Saved predictions");
    };

    println!("\nTotal runtime: {:.2?}", start.elapsed());
}
