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
