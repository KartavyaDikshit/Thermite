use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};

#[pyfunction]
fn test_allow_threads(py: Python<'_>, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
    let x_view = X.as_array();
    let y_view = y.as_array();
    py.allow_threads(|| {
        println!("Inside threads!");
        Ok(())
    })
}
