use std::ops::{Add, Sub};

/// Two-dimensional tensor implementation supporting various neural network operations
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Flattened storage of tensor elements
    pub data: Vec<f32>,
    /// Shape as (rows, columns)
    pub shape: (usize, usize),
}

impl Tensor {
    /// Creates a new tensor with default shape (1, n)
    pub fn new(data: Vec<f32>) -> Self {
        let n = data.len();
        Tensor {
            data,
            shape: (1, n),
        }
    }

    /// Creates a new tensor with specified shape
    pub fn new_with_shape(data: Vec<f32>, shape: (usize, usize)) -> Self {
        Tensor { data, shape }
    }

    /// Creates a tensor filled with zeros
    pub fn zeros(shape: (usize, usize)) -> Self {
        Tensor {
            data: vec![0.0; shape.0 * shape.1],
            shape,
        }
    }

    /// Creates a zero-filled tensor with same shape as self
    pub fn zeros_like(&self) -> Self {
        Tensor {
            data: vec![0.0; self.data.len()],
            shape: self.shape,
        }
    }

    /// Creates a tensor filled with ones
    pub fn ones(shape: (usize, usize)) -> Self {
        Tensor {
            data: vec![1.0; shape.0 * shape.1],
            shape,
        }
    }

    /// Creates a one-filled tensor with same shape as self
    pub fn ones_like(&self) -> Self {
        Tensor {
            data: vec![1.0; self.data.len()],
            shape: self.shape,
        }
    }

    /// Element-wise multiplication
    pub fn hadamard(mut self, other: &Tensor) -> Tensor {
        self.data
            .iter_mut()
            .zip(other.data.iter())
            .for_each(|(x, &y)| *x *= y);

        self
    }

    /// Matrix multiplication using optimized implementation
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let (m, n) = self.shape;
        let (_, p) = other.shape;

        let mut result = vec![0.0; m * p];

        // Using unsafe for performance optimization
        unsafe {
            for i in 0..m {
                for k in 0..n {
                    let a_val = *self.data.get_unchecked(i * n + k);
                    let row_offset = i * p;
                    let other_row_offset = k * p;

                    for j in 0..p {
                        *result.get_unchecked_mut(row_offset + j) +=
                            a_val * *other.data.get_unchecked(other_row_offset + j);
                    }
                }
            }
        }

        Tensor::new_with_shape(result, (m, p))
    }

    /// Matrix transpose operation
    pub fn transpose(&self) -> Tensor {
        let (rows, cols) = self.shape;
        let mut result = vec![0.0; self.data.len()];

        unsafe {
            for i in 0..rows {
                let row_offset = i * cols;
                for j in 0..cols {
                    *result.get_unchecked_mut(j * rows + i) =
                        *self.data.get_unchecked(row_offset + j);
                }
            }
        }

        Tensor::new_with_shape(result, (cols, rows))
    }

    /// Applies softmax function across each row
    pub fn softmax(&self) -> Tensor {
        let (rows, cols) = self.shape;
        let mut data = vec![0.0; self.data.len()];
        let mut row_sums = vec![0.0; rows];

        // Find max value per row for numerical stability
        for row in 0..rows {
            let mut max_val = f32::MIN;

            for col in 0..cols {
                let idx = row * cols + col;
                max_val = max_val.max(self.data[idx]);
            }

            // Compute exponentials and row sums
            let mut sum = 0.0;
            for col in 0..cols {
                let idx = row * cols + col;
                let exp_val = (self.data[idx] - max_val).exp();
                data[idx] = exp_val;
                sum += exp_val;
            }
            row_sums[row] = sum;
        }

        // Normalize by row sums
        for row in 0..rows {
            let sum = row_sums[row];
            for col in 0..cols {
                let idx = row * cols + col;
                data[idx] /= sum;
            }
        }

        Tensor {
            data,
            shape: self.shape,
        }
    }

    /// Converts 2D tensor shape to 4D interpretation for convolution operations
    pub fn shape4(shape: (usize, usize), channels: usize) -> (usize, usize, usize, usize) {
        let (rows, cols) = shape;
        let batch_size = rows;
        let dims = cols / channels;
        let height = (dims as f32).sqrt() as usize;
        let width = height;

        (batch_size, channels, height, width)
    }

    /// Converts image data to row format for efficient convolution
    pub fn im2row(
        &self,
        channels: usize,
        kernel_height: usize,
        kernel_width: usize,
        stride: (usize, usize),
    ) -> Tensor {
        let (batch_size, channels, height, width) = Tensor::shape4(self.shape, channels);
        let (stride_h, stride_w) = stride;

        let out_height = (height - kernel_height) / stride_h + 1;
        let out_width = (width - kernel_width) / stride_w + 1;

        let mut cols =
            vec![
                0.0;
                batch_size * out_height * out_width * channels * kernel_height * kernel_width
            ];

        // Extract patches and arrange them as rows
        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..out_height {
                    for w in 0..out_width {
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let in_h = h * stride_h + kh;
                                let in_w = w * stride_w + kw;

                                let col_idx = ((((b * out_height + h) * out_width + w) * channels
                                    + c)
                                    * kernel_height
                                    + kh)
                                    * kernel_width
                                    + kw;

                                let im_idx = ((b * channels + c) * height + in_h) * width + in_w;

                                cols[col_idx] = self.data[im_idx];
                            }
                        }
                    }
                }
            }
        }

        let out_rows = batch_size * out_height * out_width;
        let out_cols = channels * kernel_height * kernel_width;
        Tensor::new_with_shape(cols, (out_rows, out_cols))
    }

    /// Converts row format back to image format
    pub fn row2im(
        &self,
        channels: usize,
        output_shape: (usize, usize),
        kernel_height: usize,
        kernel_width: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Tensor {
        let (batch_size, channels, height, width) = Tensor::shape4(output_shape, channels);

        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = padding;

        let out_height = (height + 2 * pad_h - kernel_height) / stride_h + 1;
        let out_width = (width + 2 * pad_w - kernel_width) / stride_w + 1;

        let mut img = vec![0.0; batch_size * channels * height * width];
        let mut counts = vec![0.0; batch_size * channels * height * width];

        // Accumulate values and count contributions
        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..out_height {
                    for w in 0..out_width {
                        let col_idx = (b * out_height * out_width + h * out_width + w)
                            * (channels * kernel_height * kernel_width);

                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let im_h = h * stride_h + kh - pad_h;
                                let im_w = w * stride_w + kw - pad_w;

                                if im_h < height && im_w < width {
                                    let im_idx =
                                        ((b * channels + c) * height + im_h) * width + im_w;
                                    let col_val = self.data[c * kernel_height * kernel_width
                                        + col_idx
                                        + kh * kernel_width
                                        + kw];

                                    img[im_idx] += col_val;
                                    counts[im_idx] += 1.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::new_with_shape(img, output_shape)
    }

    /// Adds zero padding to spatial dimensions
    pub fn pad2d(&self, channels: usize, padding: (usize, usize)) -> Tensor {
        let (batch_size, channels, height, width) = Tensor::shape4(self.shape, channels);
        let (pad_h, pad_w) = padding;

        let padded_height = height + 2 * pad_h;
        let padded_width = width + 2 * pad_w;

        let mut padded = vec![0.0; batch_size * channels * padded_height * padded_width];

        // Copy data with padding
        for b in 0..batch_size {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let in_idx = ((b * channels + c) * height + h) * width + w;
                        let out_idx = ((b * channels + c) * padded_height + (h + pad_h))
                            * padded_width
                            + (w + pad_w);
                        padded[out_idx] = self.data[in_idx];
                    }
                }
            }
        }

        Tensor::new_with_shape(
            padded,
            (batch_size, channels * padded_width * padded_height),
        )
    }
}

// Element-wise addition implementation
impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, other: &'b Tensor) -> Tensor {
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();

        Tensor {
            data,
            shape: self.shape,
        }
    }
}

// Element-wise subtraction implementation
impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, other: &'b Tensor) -> Tensor {
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();

        Tensor {
            data,
            shape: self.shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{assert_close, assert_tensors_eq};

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(t.data.len(), 3);
        assert_eq!(t.shape, (1, 3));

        let t2 = Tensor::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        assert_eq!(t2.data.len(), 4);
        assert_eq!(t2.shape, (2, 2));
    }

    #[test]
    fn test_addition() {
        let a = Tensor::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let b = Tensor::new_with_shape(vec![4.0, 5.0, 6.0, 7.0], (2, 2));
        let result = &a + &b;
        assert_eq!(result.data, vec![5.0, 7.0, 9.0, 11.0]);
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn test_subtraction() {
        let a = Tensor::new_with_shape(vec![4.0, 5.0, 6.0, 7.0], (2, 2));
        let b = Tensor::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let result = &a - &b;
        assert_eq!(result.data, vec![3.0, 3.0, 3.0, 3.0]);
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn test_hadamard() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0]);
        let result = a.hadamard(&b);
        assert_eq!(result.data, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], (2, 2));
        let b = Tensor::new_with_shape(vec![5.0, 6.0, 7.0, 8.0], (2, 2));
        let result = a.matmul(&b);
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
        assert_eq!(result.shape, (2, 2));
    }

    #[test]
    fn test_transpose() {
        let a = Tensor::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
        let result = a.transpose();
        assert_eq!(result.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(result.shape, (3, 2));
    }

    #[test]
    fn test_softmax() {
        let t1 = Tensor::new_with_shape(vec![1.0, 2.0, 3.0], (1, 3));
        let s1 = t1.softmax();
        assert_close(s1.data.iter().sum::<f32>(), 1.0, 1e-6);
        assert!(s1.data.iter().all(|&x| x > 0.0 && x < 1.0));
        assert!(s1.data[2] > s1.data[1] && s1.data[1] > s1.data[0]);

        let t2 = Tensor::new_with_shape(
            vec![
                1.0, 3.0, 5.0, //
                2.0, 6.0, 4.0, //
            ],
            (2, 3),
        );
        let s2 = t2.softmax();
        let sum1 = s2.data[0] + s2.data[1] + s2.data[2];
        let sum2 = s2.data[3] + s2.data[4] + s2.data[5];
        assert_close(sum1, 1.0, 1e-6);
        assert_close(sum2, 1.0, 1e-6);

        let t3 = Tensor::new_with_shape(vec![1000.0, 1000.0], (1, 2));
        let s3 = t3.softmax();
        assert_close(s3.data[0], 0.5, 1e-6);
        assert_close(s3.data[1], 0.5, 1e-6);

        let t4 = Tensor::new_with_shape(vec![-1000.0, -1000.0], (1, 2));
        let s4 = t4.softmax();
        assert_close(s4.data[0], 0.5, 1e-6);
        assert_close(s4.data[1], 0.5, 1e-6);
    }

    #[test]
    fn test_im2row_1() {
        // One image, one channel, 4x4 input
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
            ],
            (1, 16),
        );

        let cols = input.im2row(1, 2, 2, (1, 1));

        assert_eq!(cols.data.len(), 36);
        assert_eq!(cols.shape, (9, 4));

        assert_eq!(cols.data[0..4], vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(cols.data[4..8], vec![2.0, 3.0, 6.0, 7.0]);
        assert_eq!(cols.data[8..12], vec![3.0, 4.0, 7.0, 8.0]);
        assert_eq!(cols.data[12..16], vec![5.0, 6.0, 9.0, 10.0]);
        assert_eq!(cols.data[16..20], vec![6.0, 7.0, 10.0, 11.0]);
        assert_eq!(cols.data[20..24], vec![7.0, 8.0, 11.0, 12.0]);
        assert_eq!(cols.data[24..28], vec![9.0, 10.0, 13.0, 14.0]);
        assert_eq!(cols.data[28..32], vec![10.0, 11.0, 14.0, 15.0]);
        assert_eq!(cols.data[32..36], vec![11.0, 12.0, 15.0, 16.0]);
    }

    #[test]
    fn test_im2row_2() {
        // Two images, one channel, 4x4 input
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
                ////////////////////////////
                17.0, 18.0, 19.0, 20.0, //
                21.0, 22.0, 23.0, 24.0, //
                25.0, 26.0, 27.0, 28.0, //
                29.0, 30.0, 31.0, 32.0, //
            ],
            (2, 16),
        );

        let cols = input.im2row(1, 2, 2, (1, 1));

        assert_eq!(cols.data.len(), 72);
        assert_eq!(cols.shape, (18, 4));

        assert_eq!(cols.data[0..4], vec![1.0, 2.0, 5.0, 6.0]);
        assert_eq!(cols.data[4..8], vec![2.0, 3.0, 6.0, 7.0]);
        assert_eq!(cols.data[8..12], vec![3.0, 4.0, 7.0, 8.0]);
        assert_eq!(cols.data[12..16], vec![5.0, 6.0, 9.0, 10.0]);
        assert_eq!(cols.data[16..20], vec![6.0, 7.0, 10.0, 11.0]);
        assert_eq!(cols.data[20..24], vec![7.0, 8.0, 11.0, 12.0]);
        assert_eq!(cols.data[24..28], vec![9.0, 10.0, 13.0, 14.0]);
        assert_eq!(cols.data[28..32], vec![10.0, 11.0, 14.0, 15.0]);
        assert_eq!(cols.data[32..36], vec![11.0, 12.0, 15.0, 16.0]);

        assert_eq!(cols.data[36..40], vec![17.0, 18.0, 21.0, 22.0]);
        assert_eq!(cols.data[40..44], vec![18.0, 19.0, 22.0, 23.0]);
        assert_eq!(cols.data[44..48], vec![19.0, 20.0, 23.0, 24.0]);
        assert_eq!(cols.data[48..52], vec![21.0, 22.0, 25.0, 26.0]);
        assert_eq!(cols.data[52..56], vec![22.0, 23.0, 26.0, 27.0]);
        assert_eq!(cols.data[56..60], vec![23.0, 24.0, 27.0, 28.0]);
        assert_eq!(cols.data[60..64], vec![25.0, 26.0, 29.0, 30.0]);
        assert_eq!(cols.data[64..68], vec![26.0, 27.0, 30.0, 31.0]);
        assert_eq!(cols.data[68..72], vec![27.0, 28.0, 31.0, 32.0]);
    }

    #[test]
    fn test_im2row_3() {
        // One image, two channels, 4x4 input
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, 16.0, //
                ////////////////////////////
                17.0, 18.0, 19.0, 20.0, //
                21.0, 22.0, 23.0, 24.0, //
                25.0, 26.0, 27.0, 28.0, //
                29.0, 30.0, 31.0, 32.0, //
            ],
            (1, 32),
        );

        let cols = input.im2row(2, 2, 2, (1, 1));

        assert_eq!(cols.data.len(), 72);
        assert_eq!(cols.shape, (9, 8));

        assert_eq!(
            cols.data[0..8],
            vec![1.0, 2.0, 5.0, 6.0, 17.0, 18.0, 21.0, 22.0]
        );
        assert_eq!(
            cols.data[8..16],
            vec![2.0, 3.0, 6.0, 7.0, 18.0, 19.0, 22.0, 23.0]
        );
        assert_eq!(
            cols.data[16..24],
            vec![3.0, 4.0, 7.0, 8.0, 19.0, 20.0, 23.0, 24.0]
        );
        assert_eq!(
            cols.data[24..32],
            vec![5.0, 6.0, 9.0, 10.0, 21.0, 22.0, 25.0, 26.0]
        );
        assert_eq!(
            cols.data[32..40],
            vec![6.0, 7.0, 10.0, 11.0, 22.0, 23.0, 26.0, 27.0]
        );
        assert_eq!(
            cols.data[40..48],
            vec![7.0, 8.0, 11.0, 12.0, 23.0, 24.0, 27.0, 28.0]
        );
        assert_eq!(
            cols.data[48..56],
            vec![9.0, 10.0, 13.0, 14.0, 25.0, 26.0, 29.0, 30.0]
        );
        assert_eq!(
            cols.data[56..64],
            vec![10.0, 11.0, 14.0, 15.0, 26.0, 27.0, 30.0, 31.0]
        );
        assert_eq!(
            cols.data[64..72],
            vec![11.0, 12.0, 15.0, 16.0, 27.0, 28.0, 31.0, 32.0]
        );
    }

    #[test]
    fn test_im2row_4() {
        // Two images, two channels, 3x3 input
        let input = Tensor::new_with_shape(
            vec![
                // First image, first channel
                1.0, 2.0, 3.0, //
                4.0, 5.0, 6.0, //
                7.0, 8.0, 9.0, //
                // First image, second channel
                10.0, 11.0, 12.0, //
                13.0, 14.0, 15.0, //
                16.0, 17.0, 18.0, //
                // Second image, first channel
                19.0, 20.0, 21.0, //
                22.0, 23.0, 24.0, //
                25.0, 26.0, 27.0, //
                // Second image, second channel
                28.0, 29.0, 30.0, //
                31.0, 32.0, 33.0, //
                34.0, 35.0, 36.0, //
            ],
            (2, 18),
        );

        let cols = input.im2row(2, 2, 2, (1, 1));

        assert_eq!(cols.data.len(), 64);
        assert_eq!(cols.shape, (8, 8));

        // First image, kernel 1
        assert_eq!(
            cols.data[0..8],
            vec![1.0, 2.0, 4.0, 5.0, 10.0, 11.0, 13.0, 14.0]
        );
        // First image, kernel 2
        assert_eq!(
            cols.data[8..16],
            vec![2.0, 3.0, 5.0, 6.0, 11.0, 12.0, 14.0, 15.0]
        );
        // First image, kernel 3
        assert_eq!(
            cols.data[16..24],
            vec![4.0, 5.0, 7.0, 8.0, 13.0, 14.0, 16.0, 17.0]
        );
        // First image, kernel 4
        assert_eq!(
            cols.data[24..32],
            vec![5.0, 6.0, 8.0, 9.0, 14.0, 15.0, 17.0, 18.0]
        );
        // Second image, kernel 1
        assert_eq!(
            cols.data[32..40],
            vec![19.0, 20.0, 22.0, 23.0, 28.0, 29.0, 31.0, 32.0]
        );
        // Second image, kernel 2
        assert_eq!(
            cols.data[40..48],
            vec![20.0, 21.0, 23.0, 24.0, 29.0, 30.0, 32.0, 33.0]
        );
        // Second image, kernel 3
        assert_eq!(
            cols.data[48..56],
            vec![22.0, 23.0, 25.0, 26.0, 31.0, 32.0, 34.0, 35.0]
        );
        // Second image, kernel 4
        assert_eq!(
            cols.data[56..64],
            vec![23.0, 24.0, 26.0, 27.0, 32.0, 33.0, 35.0, 36.0]
        );
    }

    #[test]
    fn test_row2im_1() {
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, 5.0, 6.0, //
                2.0, 3.0, 6.0, 7.0, //
                3.0, 4.0, 7.0, 8.0, //
                5.0, 6.0, 9.0, 10.0, //
                6.0, 7.0, 10.0, 11.0, //
                7.0, 8.0, 11.0, 12.0, //
                9.0, 10.0, 13.0, 14.0, //
                10.0, 11.0, 14.0, 15.0, //
                11.0, 12.0, 15.0, 16.0, //
            ],
            (9, 4),
        );

        let img = input.row2im(1, (1, 16), 2, 2, (1, 1), (0, 0));
        assert_eq!(img.data.len(), 16);
        assert_eq!(img.shape, (1, 16));

        assert_eq!(img.data[0..4], vec![1.0, 4.0, 6.0, 4.0]);
        assert_eq!(img.data[4..8], vec![10.0, 24.0, 28.0, 16.0]);
        assert_eq!(img.data[8..12], vec![18.0, 40.0, 44.0, 24.0]);
        assert_eq!(img.data[12..16], vec![13.0, 28.0, 30.0, 16.0]);
    }

    #[test]
    fn test_padding_1() {
        // One image, one channel, 2x2 input
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
            ],
            (1, 4),
        );

        let padded = input.pad2d(1, (1, 1));

        let expected = Tensor::new_with_shape(
            vec![
                0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 2.0, 0.0, //
                0.0, 3.0, 4.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
            ],
            (1, 16),
        );

        assert_tensors_eq(&padded, &expected);
    }

    #[test]
    fn test_padding_2() {
        // One image, two channels, 2x2 input
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                ////////////
                5.0, 6.0, //
                7.0, 8.0, //
            ],
            (1, 8),
        );

        let padded = input.pad2d(2, (1, 1));

        let expected = Tensor::new_with_shape(
            vec![
                0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 2.0, 0.0, //
                0.0, 3.0, 4.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                //////////////////////
                0.0, 0.0, 0.0, 0.0, //
                0.0, 5.0, 6.0, 0.0, //
                0.0, 7.0, 8.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
            ],
            (1, 32),
        );

        assert_tensors_eq(&padded, &expected);
    }

    #[test]
    fn test_padding_3() {
        // Two images, one channel, 2x2 input
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                ////////////
                5.0, 6.0, //
                7.0, 8.0, //
            ],
            (2, 4),
        );

        let padded = input.pad2d(1, (1, 1));

        let expected = Tensor::new_with_shape(
            vec![
                0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 2.0, 0.0, //
                0.0, 3.0, 4.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                //////////////////////
                0.0, 0.0, 0.0, 0.0, //
                0.0, 5.0, 6.0, 0.0, //
                0.0, 7.0, 8.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
            ],
            (2, 16),
        );

        assert_tensors_eq(&padded, &expected);
    }

    #[test]
    fn test_padding_4() {
        // Two images, two channels, 2x2 input
        let input = Tensor::new_with_shape(
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                ////////////
                5.0, 6.0, //
                7.0, 8.0, //
                ////////////
                9.0, 10.0, //
                11.0, 12.0, //
                ////////////
                13.0, 14.0, //
                15.0, 16.0, //
            ],
            (2, 8),
        );

        let padded = input.pad2d(2, (1, 1));

        let expected = Tensor::new_with_shape(
            vec![
                0.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 2.0, 0.0, //
                0.0, 3.0, 4.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                //////////////////////
                0.0, 0.0, 0.0, 0.0, //
                0.0, 5.0, 6.0, 0.0, //
                0.0, 7.0, 8.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                //////////////////////
                0.0, 0.0, 0.0, 0.0, //
                0.0, 9.0, 10.0, 0.0, //
                0.0, 11.0, 12.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
                //////////////////////
                0.0, 0.0, 0.0, 0.0, //
                0.0, 13.0, 14.0, 0.0, //
                0.0, 15.0, 16.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, //
            ],
            (2, 32),
        );

        assert_tensors_eq(&padded, &expected);
    }
}
