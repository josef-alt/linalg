use pyo3::prelude::{pyfunction, PyResult};

/// determinant
/// TODO dimensions check
/// TODO generics + type constraint
/// TODO allow for larger matrices
#[pyfunction]
pub fn determinant<'py>(matrix: Vec<Vec<f64>>) -> PyResult<f64> {
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

/// principal submatrix obtained by dropping row/col specified
#[pyfunction]
pub fn _extract<'py>(matrix: Vec<Vec<f64>>, row: usize, col: usize) -> PyResult<Vec<Vec<f64>>> {
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

    Ok(result)
}
