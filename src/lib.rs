use pyo3::prelude::*;
use pyo3::types::PyList;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// Converts python list to a comma separate string
/// TODO PyList deprecated
#[pyfunction]
fn list_to_string(input: &PyList) -> PyResult<String> {
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

/// A Python module implemented in Rust.
#[pymodule]
fn linalg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(list_to_string, m)?)?;
    Ok(())
}
