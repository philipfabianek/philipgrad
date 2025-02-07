use crate::Tensor;

/// Trait defining the interface for parameter optimization algorithms
pub trait Optimizer: Send + Sync {
    /// Initializes optimizer state (if any)
    fn init(&mut self, _params: &[&Tensor]) {}

    /// Updates momentum parameter if applicable,
    /// only used by Adam
    fn set_momentum(&mut self, _momentum: f32) {}

    /// Updates parameters using computed gradients
    fn update_parameters(&mut self, params: &mut Vec<(&mut Tensor, &Tensor)>, learning_rate: f32);
}

/// Basic Stochastic Gradient Descent optimizer
pub struct SGD;

impl SGD {
    pub fn new() -> Self {
        SGD
    }
}

impl Optimizer for SGD {
    fn update_parameters(&mut self, params: &mut Vec<(&mut Tensor, &Tensor)>, learning_rate: f32) {
        for (param, grad) in params.iter_mut() {
            for i in 0..param.data.len() {
                param.data[i] -= learning_rate * grad.data[i];
            }
        }
    }
}

/// Gradient Descent with Momentum
pub struct Momentum {
    /// Momentum coefficient
    momentum: f32,
    /// Velocity vectors for each parameter
    velocities: Vec<Tensor>,
}

impl Momentum {
    pub fn new(momentum: f32) -> Self {
        Momentum {
            momentum,
            velocities: Vec::new(),
        }
    }
}

impl Optimizer for Momentum {
    fn init(&mut self, params: &[&Tensor]) {
        self.velocities = params.iter().map(|p| Tensor::zeros_like(p)).collect();
    }

    fn update_parameters(&mut self, params: &mut Vec<(&mut Tensor, &Tensor)>, learning_rate: f32) {
        for ((param, grad), velocity) in params.iter_mut().zip(self.velocities.iter_mut()) {
            for i in 0..param.data.len() {
                // Update velocity and apply to parameters
                velocity.data[i] = self.momentum * velocity.data[i] - learning_rate * grad.data[i];
                param.data[i] += velocity.data[i];
            }
        }
    }
}

/// Adam optimizer (Adaptive Moment Estimation)
pub struct Adam {
    /// Exponential decay rate for first moment estimates
    beta1: f32,
    /// Exponential decay rate for second moment estimates
    beta2: f32,
    /// Small constant for numerical stability
    epsilon: f32,
    /// First moment estimates
    velocities: Vec<Tensor>,
    /// Second moment estimates
    second_moments: Vec<Tensor>,
    /// Number of update steps taken
    timestep: usize,
}

impl Adam {
    pub fn new(beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Adam {
            beta1,
            beta2,
            epsilon,
            velocities: Vec::new(),
            second_moments: Vec::new(),
            timestep: 0,
        }
    }
}

impl Optimizer for Adam {
    fn init(&mut self, params: &[&Tensor]) {
        self.velocities = params.iter().map(|p| Tensor::zeros_like(p)).collect();
        self.second_moments = params.iter().map(|p| Tensor::zeros_like(p)).collect();
    }

    fn set_momentum(&mut self, momentum: f32) {
        self.beta1 = momentum;
    }

    fn update_parameters(&mut self, params: &mut Vec<(&mut Tensor, &Tensor)>, learning_rate: f32) {
        self.timestep += 1;

        // Calculate bias correction terms
        let beta1_correction = 1.0 - self.beta1.powi(self.timestep as i32);
        let beta2_correction = 1.0 - self.beta2.powi(self.timestep as i32);
        let beta1_complement = 1.0 - self.beta1;
        let beta2_complement = 1.0 - self.beta2;

        // Update parameters using Adam algorithm
        for ((param, grad), (velocity, second_moment)) in params.iter_mut().zip(
            self.velocities
                .iter_mut()
                .zip(self.second_moments.iter_mut()),
        ) {
            param
                .data
                .iter_mut()
                .zip(grad.data.iter())
                .zip(velocity.data.iter_mut())
                .zip(second_moment.data.iter_mut())
                .for_each(
                    |(((param_val, &grad_val), velocity_val), second_moment_val)| {
                        // Update biased first moment estimate
                        *velocity_val = self.beta1 * *velocity_val + beta1_complement * grad_val;
                        // Update biased second moment estimate
                        *second_moment_val = self.beta2 * *second_moment_val
                            + beta2_complement * grad_val * grad_val;

                        // Compute bias-corrected estimates
                        let m_hat = *velocity_val / beta1_correction;
                        let v_hat = *second_moment_val / beta2_correction;

                        // Update parameters
                        *param_val -= learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                    },
                );
        }
    }
}
