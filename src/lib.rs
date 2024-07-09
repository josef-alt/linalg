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

/// determinant
/// TODO dimensions check
/// TODO generics + type constraint
/// TODO allow for larger matrices
#[pyfunction]
fn determinant<'py>(matrix: Vec<Vec<f64>>) -> PyResult<f64> {
    let mut det: f64 = 0.0;
    let n = matrix.len();
    if n == 1 {
        det = matrix[0][0];
        return Ok(det);
    }
    if n == 2 {
        det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        return Ok(det);
    }

    //     a b c   aei   afh
    // det d e f = bfg - bdi
    //     g h i   cdh   ceg
    for start in 0..n {
        let mut prod_add = 1.0;
        let mut prod_sub = 1.0;
        
        let mut r_add = 0;
        let mut c_add = start;
        
        let mut r_sub = 0;
        let mut c_sub = start;

        for _it in 0..n {
            prod_add *= matrix[r_add][c_add];
            r_add += 1;
            c_add += 1;
            if c_add == n {
                c_add = 0;
            }

            prod_sub *= matrix[r_sub][c_sub];
            r_sub += 1;
            if c_sub == 0 {
                c_sub = n - 1;
            } else {
                c_sub -= 1;
            }
        }

        //println!("diagonal + = {}", prod_add);
        //println!("diagonal - = {}", prod_sub);

        det += prod_add;
        det -= prod_sub;
    }

    Ok(det)
}

/// A Python module implemented in Rust.
#[pymodule]
fn linalg(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(list_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(scale_list, m)?)?;
    m.add_function(wrap_pyfunction!(list_dot_product, m)?)?;
    m.add_function(wrap_pyfunction!(determinant, m)?)?;
    Ok(())
}
