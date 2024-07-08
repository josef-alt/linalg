use pyo3::prelude::*;
use pyo3::types::{PyList, PyFloat};
use pyo3::exceptions::PyValueError;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// Converts python list to a comma separate string
#[pyfunction]
fn list_to_string<'py>(input: Bound<'py, PyList>) -> PyResult<String> {
    let mut total: String = "".to_owned();

    for element in input.iter() {
        total.push_str(element
            .str().expect("Conversion to PyString failed")
            .to_str().expect("Conversion to &str failed")
        );
        total.push_str(", ");
    }

    if total.len() > 0 {
        let mut chars = total.chars();
        chars.next_back();
        chars.next_back();

        total = chars.as_str().to_string();
    }

    let mut result: String = "[".to_owned();
    result.push_str(total.as_str());
    result.push_str("]");

    Ok(result)
}

/// apply scalar to vector
#[pyfunction]
fn scale_list<'py>(vector: Bound<'py, PyList>, scale: f64) -> PyResult<()> {
    for index in 0..vector.len() {
        let item = vector.get_item(index);
        let element = match item {
            Ok(obj) => obj,
            Err(e) => return Err(e),
        };

        let value = element.extract::<f64>()?;

        vector.set_item(index, PyFloat::new_bound(vector.py(), value * scale))?;
    }

    Ok(())
}

/// compute dot product of lists
#[pyfunction]
fn list_dot_product<'py>(a: Bound<'py, PyList>, b: Bound<'py, PyList>) -> PyResult<f64> {
    let mut sum: f64 = 0.0;

    let a_len = a.len();
    let b_len = b.len();
    if a_len != b_len {
        return Err(PyValueError::new_err("lists must be same length"))
    }

    for index in 0..a_len {
        let a_val = match a.get_item(index) {
            Ok(obj) => obj.extract::<f64>()?,
            Err(e) => return Err(e),
        };
        let b_val = match b.get_item(index) {
            Ok(obj) => obj.extract::<f64>()?,
            Err(e) => return Err(e),
        };
        
        sum += a_val * b_val;
    }

    Ok(sum)
}

/// A Python module implemented in Rust.
#[pymodule]
fn linalg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(list_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(scale_list, m)?)?;
    m.add_function(wrap_pyfunction!(list_dot_product, m)?)?;
    Ok(())
}
