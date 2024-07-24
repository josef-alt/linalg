use pyo3::prelude::*;

use vector::vector::{add, sub, dot_product, scale, magnitude, normalize, project};
use matrix::matrix::{determinant, identity, invert, transpose};

mod vector {
    pub mod vector;
}

mod matrix {
    pub mod matrix;
}

/// A Python module implemented in Rust.
#[pymodule]
fn linalg(m: &Bound<'_, PyModule>) -> PyResult<()> {

    // vector
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(scale, m)?)?;
    m.add_function(wrap_pyfunction!(magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(project, m)?)?;

    // matrix
    m.add_function(wrap_pyfunction!(determinant, m)?)?;
    m.add_function(wrap_pyfunction!(identity, m)?)?;
    m.add_function(wrap_pyfunction!(invert, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;

    Ok(())
}
