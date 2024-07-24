use pyo3::prelude::{pyfunction, PyResult};
use pyo3::exceptions::PyValueError;

/// generate identity matrices
#[pyfunction]
pub fn identity<'py>(size: usize) -> PyResult<Vec<Vec<i16>>> {
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
pub fn determinant<'py>(matrix: Vec<Vec<f64>>) -> PyResult<f64> {
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
pub fn is_invertible<'py>(matrix: Vec<Vec<f64>>) -> bool {
    if !_is_square(&matrix) {
        return false
    }
    return _det(&matrix) != 0.0
}

/// invert matrix or return value error
#[pyfunction]
pub fn invert<'py>(matrix: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    if !is_invertible(matrix.clone()) {
        return Err(PyValueError::new_err("matrix is not invertible"))
    }
    
    let n: usize = matrix.len();

    // compute cofactor matrix
    let mut cofactors: Vec<Vec<f64>> = Vec::new();
    for row in 0..n {
        cofactors.push(Vec::new());
        for col in 0..n {
            let minor: f64 = _det(&_extract(&matrix, row, col));
            let sign: f64 = if row % 2 == col % 2 { 1.0 } else { -1.0 };

            cofactors[row].push(sign * minor);
        }
    }
    
    // transpose cofactors
    let mut adjugate: Vec<Vec<f64>> = Vec::new();
    for row in 0..n {
        adjugate.push(Vec::new());
        for col in 0..n {
            adjugate[row].push(cofactors[col][row]);
        }
    }

    // compute determinant
    let det: f64 = _det(&matrix);

    // multiply adjudicate by 1/det
    for row in 0..n {
        for col in 0..n {
            adjugate[row][col] *= 1.0 / det;
        }
    }

    Ok(adjugate)
}
