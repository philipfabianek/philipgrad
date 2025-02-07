# PhilipGrad

A CPU-based image classifier neural network framework implemented from scratch in Rust supporting basic MLPs and CNNs.

## Context

This project was developed as part of a university assignment with the goal of implementing an image classifier neural network entirely from scratch without using any external ML libraries.

The result is a small CPU-based machine learning framework that can be used to build basic MLPs and CNNs for image classification.

The project includes several examples for classifying the Fashion MNIST datasets. One example CNN (`cnn8-16.rs`) reaches 99% accuracy on MNIST and 94% accuracy on Fashion MNIST in a reasonable time.

The assignment requirement was to reach 88% accuracy on Fashion MNIST in 10 minutes. One example (`mlp-fast.rs`) achieves this accuracy in about 5 seconds.

## Implemented Features

### Layers
- Batch Normalization
- Convolutional
- Dense
- Dropout
- LeakyReLU
- MaxPool

### Optimizers
- SGD
- Momentum
- Adam

### Learning Rate Schedulers
- Linear Decay
- Linear with Warmup
- Exponential
- Exponential with Warmup
- OneCycle

## Getting Started

### Prerequisites
- Rust

### Installation

1. Clone the repository:
```bash
git clone https://github.com/philipfabianek/philipgrad.git
cd philipgrad
```

2. Extract the dataset:
```bash
unzip data.zip
```

## Running Examples

The project contains several examples which can be run using the `run.sh` script. All examples use the Fashion MNIST dataset by default. If you want to run a different example, you can uncomment and comment lines in the file. You can also train on a different dataset, for example the included MNIST dataset.

## Implementation Details

- The framework is implemented purely in Rust with minimal dependencies (only `clap`, `rand`, and `rand_distr`)
- Most layers were implemented with a test-driven approach and hence contain reasonable tests
- In all examples 16 minibatches are processed in parallel

## License

This project is licensed under the MIT License.