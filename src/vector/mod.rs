use pyo3::prelude::{
    pymodule, PyModule,
    PyModuleMethods, PyAnyMethods,
    pyfunction, PyResult,
    Bound
};
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyValueError;

/// set up vector module functions
#[pymodule]
fn vector(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(scale, m)?)?;
    m.add_function(wrap_pyfunction!(dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(project, m)?)?;

    Ok(())
}

/// initialize module
pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(m.py(), "linalg.vector")?;
    vector(&module)?;

    m.add("vector", &module)?;
    m.py().import_bound("sys")?
        .getattr("modules")?
        .set_item("linalg.vector", module)?;

    Ok(())
}

/// component-wise addition of two same-size vectors
#[pyfunction]
fn add<'py>(a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
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
fn sub<'py>(a: Vec<f64>, b: Vec<f64>) -> PyResult<Vec<f64>> {
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
fn scale<'py>(vector: Vec<f64>, scalar: f64) -> PyResult<Vec<f64>> {
    Ok(vector.iter().map(|ele| ele * scalar).collect())
}

/// dot product
#[pyfunction]
fn dot_product<'py>(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
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
fn magnitude<'py>(vector: Vec<f64>) -> f64 {
    return _magnitude(&vector)
}

// helper function to compute magnitude without copying vector
fn _magnitude(vector: &Vec<f64>) -> f64 {
    let mut mag: f64 = 0.0;

    for ele in vector.iter() {
        mag += ele * ele;
    }
    mag = f64::sqrt(mag);

    return mag
}

// corss product

/// project vector v onto u
#[pyfunction]
fn project<'py>(u: Vec<f64>, v: Vec<f64>) -> PyResult<Vec<f64>> {
    let mag_u: f64 = _magnitude(&u);
    let dot_prod: f64 = dot_product(u.clone(), v)?;
    let scalar: f64 = dot_prod / (mag_u * mag_u);

    Ok(scale(u, scalar)?)
}

/// normalization
#[pyfunction]
fn normalize<'py>(vector: Vec<f64>) -> PyResult<Vec<f64>> {
    let norm: f64 = _magnitude(&vector);
    Ok(vector.iter().map(|ele| ele / norm).collect())
}

// decomposition
