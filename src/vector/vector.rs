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

// scalar multiplication
// dot product
// corss product
// projection
// normalization
// decomposition
