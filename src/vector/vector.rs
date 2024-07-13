use pyo3::prelude::{pyfunction, PyResult};
use pyo3::exceptions::PyValueError;

/// component-wise addition of two same-size vectors
#[pyfunction]
pub fn add<'py>(a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("vectors must be same length"))
    }

    let mut result: Vec<f64> = Vec::new();
    for idx in 0..a.len() {
        result.push(a[idx] + b[idx]);
    }

    Ok(result)
}

/// component-wise subtraction of two same-size vectors
#[pyfunction]
pub fn sub<'py>(a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("vectors must be same length"))
    }

    let mut result: Vec<f64> = Vec::new();
    for idx in 0..a.len() {
        result.push(a[idx] - b[idx]);
    }
    Ok(result)
}

/// multiply each element of a vector by a scalar
#[pyfunction]
pub fn scale<'py>(vector: Vec<f64>, scalar: f64) -> PyResult<Vec<f64>> {
    Ok(vector.iter().map(|ele| ele * scalar).collect())
}

/// dot product
#[pyfunction]
pub fn dot_product<'py>(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("vectors must be same length"))
    }

    let mut result: f64 = 0.0;
    for idx in 0..a.len() {
        result += a[idx] * b[idx];
    }
    Ok(result)
}

/// magnitude
#[pyfunction]
pub fn magnitude<'py>(vector: Vec<f64>) -> PyResult<f64> {
    let mut mag: f64 = 0.0;

    for ele in vector.iter() {
        mag += ele * ele;
    }
    mag = f64::sqrt(mag);

    Ok(mag)
}

// corss product
// projection

/// normalization
#[pyfunction]
pub fn normalize<'py>(vector: Vec<f64>) -> PyResult<Vec<f64>> {
    let norm: f64 = magnitude(vector.clone())?;
    Ok(vector.iter().map(|ele| ele / norm).collect())
}

// decomposition
