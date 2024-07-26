use pyo3::prelude::*;

mod vector;
mod matrix;

/// A Python module implemented in Rust.
#[pymodule]
fn linalg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // initializing submodules
    vector::init(m)?;
    matrix::init(m)?;
    Ok(())
}
