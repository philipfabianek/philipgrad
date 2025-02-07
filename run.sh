#########################################################################
#########################################################################

# CNN with single convolutional layer with 16 features
# - number of epochs: 20
# - expected runtime: ~3 minutes 20 seconds
# - expected epoch duration: ~9 seconds
# - expected training accuracy: ~99.70%
# - expected test accuracy: ~93.50%

cargo b --example cnn16 --release
./target/release/examples/cnn16 \
  --train-labels ./data/fashion_mnist_train_labels.csv \
  --train-vectors ./data/fashion_mnist_train_vectors.csv \
  --test-labels ./data/fashion_mnist_test_labels.csv \
  --test-vectors ./data/fashion_mnist_test_vectors.csv

#########################################################################
#########################################################################

# CNN with two convolutional layers with 8 and 16 features
# - number of epochs: 20
# - expected runtime: ~7 minutes 30 seconds
# - expected epoch duration: ~21 seconds
# - expected training accuracy: ~99.99%
# - expected test accuracy: ~94.00%

# cargo b --example cnn8-16 --release
# ./target/release/examples/cnn8-16 \
#   --train-labels ./data/fashion_mnist_train_labels.csv \
#   --train-vectors ./data/fashion_mnist_train_vectors.csv \
#   --test-labels ./data/fashion_mnist_test_labels.csv \
#   --test-vectors ./data/fashion_mnist_test_vectors.csv

#########################################################################
#########################################################################

# Minimalistic MLP with single hidden layer with 128 neurons
# - number of epochs: 3
# - expected runtime: ~7 seconds
# - expected epoch duration: ~1.3 second
# - expected training accuracy: ~89.50%
# - expected test accuracy: ~88.00%

# cargo b --example mlp-fast --release
# ./target/release/examples/mlp-fast \
#   --train-labels ./data/fashion_mnist_train_labels.csv \
#   --train-vectors ./data/fashion_mnist_train_vectors.csv \
#   --test-labels ./data/fashion_mnist_test_labels.csv \
#   --test-vectors ./data/fashion_mnist_test_vectors.csv

#########################################################################
#########################################################################

# MLP with two hidden layers with 512 and 256 neurons
# - number of epochs: 25
# - expected runtime: ~80 seconds
# - expected epoch duration: ~3 seconds
# - expected training accuracy: ~98.00%
# - expected test accuracy: ~91.50%

# cargo b --example mlp --release
# ./target/release/examples/mlp \
#   --train-labels ./data/fashion_mnist_train_labels.csv \
#   --train-vectors ./data/fashion_mnist_train_vectors.csv \
#   --test-labels ./data/fashion_mnist_test_labels.csv \
#   --test-vectors ./data/fashion_mnist_test_vectors.csv