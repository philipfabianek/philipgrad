/// Learning rate scheduler implementations for different update strategies
#[derive(Debug)]
pub enum LearningRateScheduler {
    /// Linear decay from max_lr to max_lr/final_div_factor
    Linear {
        /// Maximum learning rate
        max_lr: f32,
        /// Factor to divide max_lr by to get final learning rate
        final_div_factor: f32,
    },

    /// Linear schedule with warmup from max_lr/div_factor to max_lr, then decay to final_lr
    LinearWithWarmup {
        /// Maximum learning rate
        max_lr: f32,
        /// Factor to divide max_lr by to get initial learning rate
        div_factor: f32,
        /// Factor to divide initial_lr by to get final learning rate
        final_div_factor: f32,
        /// Percentage of training spent in warmup phase
        pct_start: f32,
    },

    /// Exponential decay from max_lr to max_lr/final_div_factor
    Exponential {
        /// Maximum learning rate
        max_lr: f32,
        /// Factor to divide max_lr by to get final learning rate
        final_div_factor: f32,
    },

    /// Exponential warmup and decay schedule
    ExponentialWithWarmup {
        /// Maximum learning rate
        max_lr: f32,
        /// Factor to divide max_lr by to get initial learning rate
        div_factor: f32,
        /// Factor to divide initial_lr by to get final learning rate
        final_div_factor: f32,
        /// Percentage of training spent in warmup phase
        pct_start: f32,
    },

    /// One-Cycle learning rate policy with cosine annealing,
    /// uses both learning rate and momentum scheduling
    OneCycle {
        /// Maximum learning rate at peak of cycle
        max_lr: f32,
        /// Factor to divide max_lr by to get initial learning rate
        div_factor: f32,
        /// Factor to divide initial_lr by to get final learning rate
        final_div_factor: f32,
        /// Percentage of training spent in ascent to max_lr
        pct_start: f32,
        /// Base (minimum) momentum value
        base_momentum: f32,
        /// Maximum momentum value
        max_momentum: f32,
    },
}

/// Output from scheduler containing updated parameters
#[derive(Debug)]
pub enum SchedulerOutput {
    /// Basic output with just learning rate
    Basic { learning_rate: f32 },
    /// Output with both learning rate and momentum,
    /// used only by OneCycle
    WithMomentum { learning_rate: f32, momentum: f32 },
}

impl LearningRateScheduler {
    /// Calculates learning rate (and momentum if applicable) for current training step
    ///
    /// # Arguments
    /// * `current_step` - Current training step
    /// * `total_steps` - Total number of training steps
    ///
    /// # Returns
    /// SchedulerOutput containing updated parameters
    pub fn get_lr(
        self: &LearningRateScheduler,
        current_step: usize,
        total_steps: usize,
    ) -> SchedulerOutput {
        match self {
            LearningRateScheduler::Linear {
                max_lr,
                final_div_factor,
            } => {
                let initial_lr = max_lr;
                let final_lr = max_lr / final_div_factor;
                let progress = current_step as f32 / total_steps as f32;
                // Linear interpolation between initial and final lr
                let lr = initial_lr + (final_lr - initial_lr) * progress;

                SchedulerOutput::Basic { learning_rate: lr }
            }
            LearningRateScheduler::LinearWithWarmup {
                max_lr,
                div_factor,
                final_div_factor,
                pct_start,
            } => {
                let initial_lr = max_lr / div_factor;
                let final_lr = initial_lr / final_div_factor;
                let steps_to_max = (total_steps as f32 * pct_start) as usize;

                let lr = if current_step < steps_to_max {
                    // Linear warmup phase
                    let progress = current_step as f32 / steps_to_max as f32;
                    initial_lr + (max_lr - initial_lr) * progress
                } else {
                    // Linear decay phase
                    let progress =
                        (current_step - steps_to_max) as f32 / (total_steps - steps_to_max) as f32;
                    max_lr + (final_lr - max_lr) * progress
                };

                SchedulerOutput::Basic { learning_rate: lr }
            }
            LearningRateScheduler::Exponential {
                max_lr,
                final_div_factor,
            } => {
                let initial_lr = max_lr;
                let final_lr = max_lr / final_div_factor;
                // Calculate decay rate to reach final_lr in total_steps
                let decay_rate = (final_lr / initial_lr).powf(1.0 / total_steps as f32);
                let lr = initial_lr * decay_rate.powf(current_step as f32);

                SchedulerOutput::Basic { learning_rate: lr }
            }
            LearningRateScheduler::ExponentialWithWarmup {
                max_lr,
                div_factor,
                final_div_factor,
                pct_start,
            } => {
                let initial_lr = max_lr / div_factor;
                let final_lr = initial_lr / final_div_factor;
                let steps_to_max = (total_steps as f32 * pct_start) as usize;

                let lr = if current_step < steps_to_max {
                    // Exponential warmup phase
                    let decay_rate = (max_lr / initial_lr).powf(1.0 / steps_to_max as f32);
                    initial_lr * decay_rate.powf(current_step as f32)
                } else {
                    // Exponential decay phase
                    let adjusted_step = current_step - steps_to_max;
                    let remaining_steps = total_steps - steps_to_max;
                    let decay_rate = (final_lr / max_lr).powf(1.0 / remaining_steps as f32);
                    max_lr * decay_rate.powf(adjusted_step as f32)
                };

                SchedulerOutput::Basic { learning_rate: lr }
            }
            LearningRateScheduler::OneCycle {
                max_lr,
                div_factor,
                final_div_factor,
                pct_start,
                base_momentum,
                max_momentum,
            } => {
                let initial_lr = max_lr / div_factor;
                let final_lr = initial_lr / final_div_factor;
                let steps_to_max = (total_steps as f32 * pct_start) as usize;

                // Calculate learning rate using cosine annealing
                let lr = if current_step < steps_to_max {
                    let progress = current_step as f32 / steps_to_max as f32;
                    let cos_out = (progress * std::f32::consts::PI).cos() + 1.0;
                    initial_lr + (max_lr - initial_lr) * (1.0 - cos_out / 2.0)
                } else {
                    let progress =
                        (current_step - steps_to_max) as f32 / (total_steps - steps_to_max) as f32;
                    let cos_out = (progress * std::f32::consts::PI).cos() + 1.0;
                    max_lr + (final_lr - max_lr) * (1.0 - cos_out / 2.0)
                };

                // Calculate momentum using inverse schedule
                let momentum = if current_step < steps_to_max {
                    let progress = current_step as f32 / steps_to_max as f32;
                    let cos_out = (progress * std::f32::consts::PI).cos() + 1.0;
                    max_momentum + (base_momentum - max_momentum) * (1.0 - cos_out / 2.0)
                } else {
                    let progress =
                        (current_step - steps_to_max) as f32 / (total_steps - steps_to_max) as f32;
                    let cos_out = (progress * std::f32::consts::PI).cos() + 1.0;
                    base_momentum + (max_momentum - base_momentum) * (1.0 - cos_out / 2.0)
                };

                SchedulerOutput::WithMomentum {
                    learning_rate: lr,
                    momentum,
                }
            }
        }
    }
}
