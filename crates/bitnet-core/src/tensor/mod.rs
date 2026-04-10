//! Tensor type for BitNet b1.58 inference.
//!
//! # Design Philosophy
//!
//! [`Tensor<T>`] is a lightweight, heap-backed multi-dimensional array with a
//! fixed rank of up to 4 dimensions.  It is deliberately simple:
//!
//! - **No autograd** — inference only.
//! - **No device abstraction** — `Tensor<T>` always lives on the host.
//!   GPU/NPU backends receive raw slices and manage their own memory.
//! - **Row-major (C-contiguous) layout** — elements are stored in the order
//!   `[d0, d1, d2, d3]` with the last index varying fastest.
//!
//! # Shape Convention
//!
//! Unused trailing dimensions are set to `1`.  A 1-D vector of length `n` has
//! shape `[n, 1, 1, 1]`.  A 2-D matrix `[rows, cols]` has shape
//! `[rows, cols, 1, 1]`.
//!
//! # Strides
//!
//! Strides are computed automatically from the shape and are always contiguous:
//!
//! ```text
//! stride[3] = 1
//! stride[2] = shape[3]
//! stride[1] = shape[3] * shape[2]
//! stride[0] = shape[3] * shape[2] * shape[1]
//! ```
//!
//! This matches NumPy C-order strides (in elements, not bytes).

use std::ops::{Index, IndexMut};

use crate::error::{BitNetError, Result};

pub mod dtype;
pub use dtype::DType;

// ---------------------------------------------------------------------------
// Tensor<T>
// ---------------------------------------------------------------------------

/// A heap-backed, row-major, up-to-4-D array.
///
/// # Type Parameter
/// `T` must be `Clone + Default` so that tensors can be zero-initialised and
/// reshaped.  Typical types: `f32`, `f16`, `bf16`, `i8`, `u8`.
///
/// # Memory Layout
/// Elements are stored in a flat `Vec<T>` in row-major order.
/// The total number of elements is `shape[0] * shape[1] * shape[2] * shape[3]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T: Clone + Default> {
    /// Flat, row-major element storage.
    data: Vec<T>,
    /// Shape in each of the 4 dimensions.  Unused dims are `1`.
    shape: [usize; 4],
    /// Stride in elements for each dimension (always contiguous).
    strides: [usize; 4],
}

impl<T: Clone + Default> Tensor<T> {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Create a tensor filled with the default value of `T` (usually zero).
    ///
    /// # Errors
    /// Returns [`BitNetError::InvalidShape`] if any shape dimension is 0, or
    /// if the total number of elements overflows `usize`.
    pub fn zeros(shape: [usize; 4]) -> Result<Self> {
        Self::validate_shape(&shape)?;
        let numel = shape.iter().product::<usize>();
        Ok(Self {
            data: vec![T::default(); numel],
            strides: Self::compute_strides(&shape),
            shape,
        })
    }

    /// Create a tensor from an existing `Vec<T>` and a shape.
    ///
    /// # Errors
    /// Returns [`BitNetError::InvalidShape`] if `data.len()` does not equal
    /// the product of the shape dimensions, or if any dimension is 0.
    pub fn from_vec(data: Vec<T>, shape: [usize; 4]) -> Result<Self> {
        Self::validate_shape(&shape)?;
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(BitNetError::shape(
                format!("shape {:?} requires {} elements", shape, numel),
                format!("{} elements in data", data.len()),
            ));
        }
        Ok(Self {
            strides: Self::compute_strides(&shape),
            data,
            shape,
        })
    }

    /// Create a 1-D tensor (vector) from a `Vec<T>`.
    ///
    /// Shape is `[n, 1, 1, 1]` where `n = data.len()`.
    ///
    /// # Errors
    /// Returns [`BitNetError::InvalidShape`] if `data` is empty.
    pub fn from_vec_1d(data: Vec<T>) -> Result<Self> {
        let n = data.len();
        if n == 0 {
            return Err(BitNetError::shape("non-empty 1-D tensor", "0 elements"));
        }
        Self::from_vec(data, [n, 1, 1, 1])
    }

    /// Create a 2-D tensor (matrix) from a `Vec<T>`.
    ///
    /// Shape is `[rows, cols, 1, 1]`.
    ///
    /// # Errors
    /// Returns [`BitNetError::InvalidShape`] if `data.len() != rows * cols` or
    /// if either dimension is 0.
    pub fn from_vec_2d(data: Vec<T>, rows: usize, cols: usize) -> Result<Self> {
        Self::from_vec(data, [rows, cols, 1, 1])
    }

    // ------------------------------------------------------------------
    // Shape / metadata accessors
    // ------------------------------------------------------------------

    /// Returns the shape array `[d0, d1, d2, d3]`.
    #[inline]
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }

    /// Returns the strides array `[s0, s1, s2, s3]`.
    #[inline]
    pub fn strides(&self) -> [usize; 4] {
        self.strides
    }

    /// Total number of elements: `d0 * d1 * d2 * d3`.
    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the tensor has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Number of dimensions (rank) ignoring trailing `1`s.
    ///
    /// A scalar (all dims = 1) has rank 0.
    /// A vector `[n, 1, 1, 1]` has rank 1.
    /// A matrix `[r, c, 1, 1]` has rank 2.
    pub fn rank(&self) -> usize {
        let mut r = 4;
        while r > 0 && self.shape[r - 1] == 1 {
            r -= 1;
        }
        r
    }

    // ------------------------------------------------------------------
    // Data access
    // ------------------------------------------------------------------

    /// Returns a shared reference to the flat data slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable reference to the flat data slice.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns a shared reference to the underlying `Vec<T>`.
    #[inline]
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Consume the tensor, returning the flat `Vec<T>`.
    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Compute the flat index for a 4-D coordinate `[i0, i1, i2, i3]`.
    ///
    /// # Errors
    /// Returns [`BitNetError::InvalidShape`] if any index is out of bounds.
    #[inline]
    pub fn flat_index(&self, idx: [usize; 4]) -> Result<usize> {
        for dim in 0..4 {
            if idx[dim] >= self.shape[dim] {
                return Err(BitNetError::shape(
                    format!("index[{dim}] < shape[{dim}] = {}", self.shape[dim]),
                    format!("index[{dim}] = {}", idx[dim]),
                ));
            }
        }
        Ok(idx[0] * self.strides[0]
            + idx[1] * self.strides[1]
            + idx[2] * self.strides[2]
            + idx[3] * self.strides[3])
    }

    /// Get an element by 4-D index.
    ///
    /// # Errors
    /// Returns [`BitNetError::InvalidShape`] if any index is out of bounds.
    pub fn get(&self, idx: [usize; 4]) -> Result<&T> {
        let flat = self.flat_index(idx)?;
        Ok(&self.data[flat])
    }

    /// Get a mutable reference to an element by 4-D index.
    ///
    /// # Errors
    /// Returns [`BitNetError::InvalidShape`] if any index is out of bounds.
    pub fn get_mut(&mut self, idx: [usize; 4]) -> Result<&mut T> {
        let flat = self.flat_index(idx)?;
        Ok(&mut self.data[flat])
    }

    // ------------------------------------------------------------------
    // Reshape
    // ------------------------------------------------------------------

    /// Return a new tensor with the same data but a different shape.
    ///
    /// The product of the new shape must equal `self.numel()`.
    ///
    /// # Errors
    /// Returns [`BitNetError::InvalidShape`] if `numel` mismatch or any dim = 0.
    pub fn reshape(&self, new_shape: [usize; 4]) -> Result<Self> {
        Self::validate_shape(&new_shape)?;
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(BitNetError::shape(
                format!(
                    "reshape target shape {:?} = {} elements",
                    new_shape, new_numel
                ),
                format!("tensor has {} elements", self.numel()),
            ));
        }
        Ok(Self {
            data: self.data.clone(),
            strides: Self::compute_strides(&new_shape),
            shape: new_shape,
        })
    }

    // ------------------------------------------------------------------
    // Views and slicing
    // ------------------------------------------------------------------

    /// Return an immutable [`TensorView`] of the entire tensor.
    #[inline]
    pub fn view(&self) -> TensorView<'_, T> {
        TensorView {
            data: &self.data,
            shape: self.shape,
            strides: self.strides,
        }
    }

    /// Return a contiguous slice of dimension 0 (row slice for 2-D tensors).
    ///
    /// For a matrix `[rows, cols, 1, 1]`, `row_slice(r)` returns the `r`-th
    /// row as a `&[T]` of length `cols`.
    ///
    /// # Errors
    /// Returns [`BitNetError::InvalidShape`] if `row >= shape[0]`.
    pub fn row_slice(&self, row: usize) -> Result<&[T]> {
        if row >= self.shape[0] {
            return Err(BitNetError::shape(
                format!("row < shape[0] = {}", self.shape[0]),
                format!("row = {row}"),
            ));
        }
        let stride0 = self.strides[0];
        let start = row * stride0;
        Ok(&self.data[start..start + stride0])
    }

    // ------------------------------------------------------------------
    // Fill helpers
    // ------------------------------------------------------------------

    /// Fill all elements with `value`.
    pub fn fill(&mut self, value: T) {
        for v in self.data.iter_mut() {
            *v = value.clone();
        }
    }

    /// Zero all elements (set to `T::default()`).
    pub fn zero(&mut self) {
        self.fill(T::default());
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn validate_shape(shape: &[usize; 4]) -> Result<()> {
        for (i, &d) in shape.iter().enumerate() {
            if d == 0 {
                return Err(BitNetError::shape(
                    format!("all shape dims > 0"),
                    format!("shape[{i}] = 0"),
                ));
            }
        }
        // Check for overflow.
        let mut numel: usize = 1;
        for &d in shape.iter() {
            numel = numel
                .checked_mul(d)
                .ok_or_else(|| BitNetError::shape("shape product fits in usize", "overflow"))?;
        }
        Ok(())
    }

    #[inline]
    fn compute_strides(shape: &[usize; 4]) -> [usize; 4] {
        let s3 = 1_usize;
        let s2 = shape[3] * s3;
        let s1 = shape[2] * s2;
        let s0 = shape[1] * s1;
        [s0, s1, s2, s3]
    }
}

// ---------------------------------------------------------------------------
// Index / IndexMut for flat (1-D) access
// ---------------------------------------------------------------------------

impl<T: Clone + Default> Index<usize> for Tensor<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T: Clone + Default> IndexMut<usize> for Tensor<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

// ---------------------------------------------------------------------------
// TensorView<'_, T>
// ---------------------------------------------------------------------------

/// An immutable, non-owning view into a [`Tensor<T>`] (or any `&[T]`).
///
/// Useful for passing sub-tensors to backend operations without copying.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a, T> {
    /// Shared reference to the underlying data slice.
    pub data: &'a [T],
    /// Shape of this view.
    pub shape: [usize; 4],
    /// Strides for this view (in elements).
    pub strides: [usize; 4],
}

impl<'a, T: Clone + Default> TensorView<'a, T> {
    /// Create a view from a slice and a shape.
    ///
    /// # Errors
    /// Returns [`BitNetError::InvalidShape`] if `data.len() != shape product`.
    pub fn from_slice(data: &'a [T], shape: [usize; 4]) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(BitNetError::shape(
                format!("shape {:?} = {} elements", shape, numel),
                format!("{} elements in slice", data.len()),
            ));
        }
        Ok(Self {
            data,
            strides: Tensor::<T>::compute_strides(&shape),
            shape,
        })
    }

    /// Total number of elements in this view.
    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Access a row (dimension-0 slice) for 2-D views.
    ///
    /// Returns `&data[row * strides[0] .. (row+1) * strides[0]]`.
    ///
    /// # Errors
    /// Returns [`BitNetError::InvalidShape`] if `row >= shape[0]`.
    pub fn row_slice(&self, row: usize) -> Result<&[T]> {
        if row >= self.shape[0] {
            return Err(BitNetError::shape(
                format!("row < shape[0] = {}", self.shape[0]),
                format!("row = {row}"),
            ));
        }
        let s0 = self.strides[0];
        Ok(&self.data[row * s0..(row + 1) * s0])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    #[test]
    fn zeros_creates_correct_size() {
        let t: Tensor<f32> = Tensor::zeros([2, 3, 1, 1]).unwrap();
        assert_eq!(t.numel(), 6);
        assert!(t.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn from_vec_2d_shape_and_strides() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = Tensor::from_vec_2d(data.clone(), 3, 4).unwrap();
        assert_eq!(t.shape(), [3, 4, 1, 1]);
        assert_eq!(t.strides(), [4, 1, 1, 1]);
        assert_eq!(t.numel(), 12);
        assert_eq!(t.as_slice(), data.as_slice());
    }

    #[test]
    fn from_vec_1d() {
        let t = Tensor::<i8>::from_vec_1d(vec![1, -1, 0, 1]).unwrap();
        assert_eq!(t.shape(), [4, 1, 1, 1]);
        assert_eq!(t.rank(), 1);
    }

    #[test]
    fn zeros_shape_with_zero_dim_returns_error() {
        let result: Result<Tensor<f32>> = Tensor::zeros([3, 0, 1, 1]);
        assert!(matches!(result, Err(BitNetError::InvalidShape { .. })));
    }

    #[test]
    fn from_vec_wrong_size_returns_error() {
        let data = vec![0.0_f32; 5];
        let result = Tensor::from_vec(data, [2, 3, 1, 1]); // need 6 elements
        assert!(matches!(result, Err(BitNetError::InvalidShape { .. })));
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    #[test]
    fn flat_index_2d() {
        // shape [3, 4], row-major: element [r, c] is at r*4 + c
        let t: Tensor<f32> = Tensor::zeros([3, 4, 1, 1]).unwrap();
        assert_eq!(t.flat_index([0, 0, 0, 0]).unwrap(), 0);
        assert_eq!(t.flat_index([1, 2, 0, 0]).unwrap(), 6); // 1*4 + 2
        assert_eq!(t.flat_index([2, 3, 0, 0]).unwrap(), 11); // 2*4 + 3
    }

    #[test]
    fn flat_index_out_of_bounds_returns_error() {
        let t: Tensor<f32> = Tensor::zeros([2, 3, 1, 1]).unwrap();
        let err = t.flat_index([2, 0, 0, 0]).unwrap_err(); // row 2 is OOB for shape[0]=2
        assert!(matches!(err, BitNetError::InvalidShape { .. }));
    }

    #[test]
    fn get_and_get_mut() {
        let mut t = Tensor::from_vec_2d(vec![1.0_f32, 2.0, 3.0, 4.0], 2, 2).unwrap();
        assert_eq!(*t.get([0, 1, 0, 0]).unwrap(), 2.0);
        *t.get_mut([1, 0, 0, 0]).unwrap() = 99.0;
        assert_eq!(*t.get([1, 0, 0, 0]).unwrap(), 99.0);
    }

    #[test]
    fn index_operator() {
        let t = Tensor::<f32>::from_vec_1d(vec![10.0, 20.0, 30.0]).unwrap();
        assert_eq!(t[0], 10.0);
        assert_eq!(t[2], 30.0);
    }

    #[test]
    fn index_mut_operator() {
        let mut t = Tensor::<i8>::from_vec_1d(vec![1, 0, -1]).unwrap();
        t[1] = 127;
        assert_eq!(t[1], 127);
    }

    // ------------------------------------------------------------------
    // Row slice
    // ------------------------------------------------------------------

    #[test]
    fn row_slice_2d() {
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let t = Tensor::from_vec_2d(data, 2, 3).unwrap();
        assert_eq!(t.row_slice(0).unwrap(), &[0.0_f32, 1.0, 2.0]);
        assert_eq!(t.row_slice(1).unwrap(), &[3.0_f32, 4.0, 5.0]);
    }

    #[test]
    fn row_slice_out_of_bounds_returns_error() {
        let t: Tensor<f32> = Tensor::zeros([2, 4, 1, 1]).unwrap();
        assert!(t.row_slice(2).is_err());
    }

    // ------------------------------------------------------------------
    // Reshape
    // ------------------------------------------------------------------

    #[test]
    fn reshape_preserves_data() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = Tensor::from_vec_2d(data.clone(), 3, 4).unwrap();
        let t2 = t.reshape([4, 3, 1, 1]).unwrap();
        assert_eq!(t2.as_slice(), data.as_slice());
        assert_eq!(t2.shape(), [4, 3, 1, 1]);
    }

    #[test]
    fn reshape_wrong_numel_returns_error() {
        let t: Tensor<f32> = Tensor::zeros([2, 6, 1, 1]).unwrap(); // 12 elements
        assert!(t.reshape([3, 5, 1, 1]).is_err()); // 15 elements — mismatch
    }

    // ------------------------------------------------------------------
    // Fill / zero
    // ------------------------------------------------------------------

    #[test]
    fn fill_and_zero() {
        let mut t: Tensor<f32> = Tensor::zeros([2, 3, 1, 1]).unwrap();
        t.fill(7.5);
        assert!(t.as_slice().iter().all(|&v| v == 7.5));
        t.zero();
        assert!(t.as_slice().iter().all(|&v| v == 0.0));
    }

    // ------------------------------------------------------------------
    // Rank
    // ------------------------------------------------------------------

    #[test]
    fn rank_of_vector() {
        let t: Tensor<f32> = Tensor::zeros([10, 1, 1, 1]).unwrap();
        assert_eq!(t.rank(), 1);
    }

    #[test]
    fn rank_of_matrix() {
        let t: Tensor<f32> = Tensor::zeros([5, 4, 1, 1]).unwrap();
        assert_eq!(t.rank(), 2);
    }

    #[test]
    fn rank_of_4d_tensor() {
        let t: Tensor<f32> = Tensor::zeros([2, 3, 4, 5]).unwrap();
        assert_eq!(t.rank(), 4);
    }

    // ------------------------------------------------------------------
    // TensorView
    // ------------------------------------------------------------------

    #[test]
    fn view_from_tensor() {
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let t = Tensor::from_vec_2d(data, 2, 3).unwrap();
        let v = t.view();
        assert_eq!(v.shape, [2, 3, 1, 1]);
        assert_eq!(v.numel(), 6);
    }

    #[test]
    fn tensor_view_row_slice() {
        let data: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let view = TensorView::<i8>::from_slice(&data, [2, 3, 1, 1]).unwrap();
        assert_eq!(view.row_slice(0).unwrap(), &[1i8, 2, 3]);
        assert_eq!(view.row_slice(1).unwrap(), &[4i8, 5, 6]);
    }

    #[test]
    fn tensor_view_wrong_size_returns_error() {
        let data = vec![0.0_f32; 5];
        assert!(TensorView::from_slice(&data, [2, 3, 1, 1]).is_err());
    }

    // ------------------------------------------------------------------
    // into_vec / data
    // ------------------------------------------------------------------

    #[test]
    fn into_vec_recovers_data() {
        let data = vec![1.0_f32, 2.0, 3.0];
        let t = Tensor::from_vec_1d(data.clone()).unwrap();
        assert_eq!(t.into_vec(), data);
    }
}
