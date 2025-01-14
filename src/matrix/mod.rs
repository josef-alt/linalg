use pyo3::prelude::{
    pymodule, PyModule,
    PyModuleMethods, PyAnyMethods,
    pyfunction, PyResult,
    Bound
};
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyValueError;

use rayon::prelude::*;

/// set up matrix module functions
#[pymodule]
fn matrix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(identity, m)?)?;
    m.add_function(wrap_pyfunction!(determinant, m)?)?;
    m.add_function(wrap_pyfunction!(transpose, m)?)?;
    m.add_function(wrap_pyfunction!(invert, m)?)?;

	m.add_function(wrap_pyfunction!(det_p, m)?)?;

    Ok(())
}

/// initialize submodule
pub fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new_bound(m.py(), "linalg.matrix")?;
    matrix(&module)?;

    m.add("matrix", &module)?;
    m.py().import_bound("sys")?
        .getattr("modules")?
        .set_item("linalg.matrix", module)?;

    Ok(())
}

/// generate identity matrices
#[pyfunction]
fn identity<'py>(size: usize) -> PyResult<Vec<Vec<i16>>> {
    let mut result: Vec<Vec<i16>> = Vec::new();

    for r in 0..size {
        result.push(Vec::new());
        for c in 0..size {
            if r == c {
                result[r].push(1);
            } else {
                result[r].push(0);
            }
        }
    }

    Ok(result)
}

/// determinant
/// TODO generics + type constraint
#[pyfunction]
fn determinant<'py>(matrix: Vec<Vec<f64>>) -> PyResult<f64> {
    if _is_square(&matrix) {
        return Ok(_det(&matrix))
    }
    Err(PyValueError::new_err("matrix must be square"))
}

/// helper function for computing determinant of a square matrix
fn _det<'py>(matrix: &Vec<Vec<f64>>) -> f64 {
    let n = matrix.len();
    if n == 1 {
        return matrix[0][0]
    }
    if n == 2 {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    }

    let mut det: f64 = 0.0;
    for i in 0..n {
        if matrix[0][i] != 0.0 {
            let mut minor: f64 = _det(&_extract(&matrix, 0, i));
            if i % 2 == 1 {
                minor = -minor;
            }

            det += matrix[0][i] * minor;
        }
    }

    return det
}

/// submatrix obtained by dropping row/col specified
fn _extract(matrix: &Vec<Vec<f64>>, row: usize, col: usize) -> Vec<Vec<f64>> {
    let mut result: Vec<Vec<f64>> = Vec::new();

    for r in 0..matrix.len() {
        if r != row {
            let mut new_row: Vec<f64> = Vec::new();

            for c in 0..matrix.len() {
                if c != col {
                    new_row.push(matrix[r][c]);
                }
            }

            result.push(new_row);
        }
    }

    return result
}

/// attempting to parallelize determinant
#[pyfunction]
fn det_p<'py>(matrix: Vec<Vec<f64>>) -> PyResult<f64> {
    if _is_square(&matrix) {
        return Ok(_det_p(&matrix))
    }
    Err(PyValueError::new_err("matrix must be square"))
}

fn _det_p<'py>(matrix: &Vec<Vec<f64>>) -> f64 {
    let n = matrix.len();
    if n == 1 {
        return matrix[0][0]
    }
    if n == 2 {
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    }

	let mut det: f64 = (0..n).into_par_iter().map(|i| {
		if matrix[0][i] != 0.0 {
            let mut minor: f64 = _det(&_extract(&matrix, 0, i));
            if i % 2 == 1 {
                minor = -minor;
            }

			matrix[0][i] * minor
        } else {
			0.0
		}
	}).sum();

    return det
}

/// determine if matrix is square
/// many unary operations only work on square matrices
/// this will probably be used internally only
fn _is_square(matrix: &Vec<Vec<f64>>) -> bool {
    let size: usize = matrix.len();

    for row in 0..size {
        if matrix[row].len() != size {
            return false
        }
    }

    return true
}

/// determine if two matrices are same size
/// most binary operations require the same dimensions
fn _same_size(matrix_a: &Vec<Vec<f64>>, matrix_b: &Vec<Vec<f64>>) -> bool {
    let row_count: usize = matrix_a.len();
    if matrix_b.len() != row_count {
        return false
    }
    
    for row in 0..row_count {
        if matrix_a[row].len() != matrix_b[row].len() {
            return false
        }
    }

    return true
}

/// determine whether or not a matrix is invertible
#[pyfunction]
fn is_invertible<'py>(matrix: Vec<Vec<f64>>) -> bool {
    if !_is_square(&matrix) {
        return false
    }
    return _det(&matrix) != 0.0
}

/// transpose matrix
#[pyfunction]
fn transpose<'py>(matrix: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    _transpose(&matrix)    
}

/// extracted transposition logic to allow internal use without copying matrices
fn _transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut result: Vec<Vec<f64>> = Vec::new();

    for r in 0..matrix.len() {
        for c in 0..matrix[r].len() {
            if c >= result.len() {
                result.push(Vec::new());
            }
            result[c].push(matrix[r][c]);
        }
    }

    return result
}

/// invert matrix or return value error
#[pyfunction]
fn invert<'py>(matrix: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    let n: usize = matrix.len();

    // compute cofactor matrix
    let mut cofactors: Vec<Vec<f64>> = Vec::new();
    for row in 0..n {
        // make sure matrix is square
        if matrix[row].len() != n {
            return Err(PyValueError::new_err("matrix is not invertible"))
        }

        cofactors.push(Vec::new());
        for col in 0..n {
            let minor: f64 = _det(&_extract(&matrix, row, col));
            let sign: f64 = if row % 2 == col % 2 { 1.0 } else { -1.0 };

            cofactors[row].push(sign * minor);
        }
    }
    
    // transpose cofactors
    let mut adjugate: Vec<Vec<f64>> = _transpose(&cofactors);

    // compute determinant without recomputing minors
    let mut det: f64 = 0.0;
    for col in 0..n {
        det += matrix[0][col] * cofactors[0][col];
    }

    if det == 0.0 {
        return Err(PyValueError::new_err("matrix is not invertible"))
    }

    // multiply adjudicate by 1/det
    for row in 0..n {
        for col in 0..n {
            adjugate[row][col] *= 1.0 / det;
        }
    }

    Ok(adjugate)
}
