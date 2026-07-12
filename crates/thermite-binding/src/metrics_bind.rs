use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use thermite_core::metrics as core_metrics;

#[pyfunction]
pub fn accuracy_score(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>) -> PyResult<f64> {
    core_metrics::accuracy_score(y_true.as_slice()?, y_pred.as_slice()?)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="binary"))]
pub fn precision_score(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>, average: &str) -> PyResult<f64> {
    core_metrics::precision_score(y_true.as_slice()?, y_pred.as_slice()?, average)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="binary"))]
pub fn recall_score(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>, average: &str) -> PyResult<f64> {
    core_metrics::recall_score(y_true.as_slice()?, y_pred.as_slice()?, average)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (y_true, y_pred, average="binary"))]
pub fn f1_score(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>, average: &str) -> PyResult<f64> {
    core_metrics::f1_score(y_true.as_slice()?, y_pred.as_slice()?, average)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
pub fn roc_auc_score(y_true: PyReadonlyArray1<f64>, y_score: PyReadonlyArray1<f64>) -> PyResult<f64> {
    core_metrics::roc_auc_score(y_true.as_slice()?, y_score.as_slice()?)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
pub fn mean_squared_error(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>) -> PyResult<f64> {
    core_metrics::mean_squared_error(y_true.as_slice()?, y_pred.as_slice()?)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
pub fn r2_score(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>) -> PyResult<f64> {
    core_metrics::r2_score(y_true.as_slice()?, y_pred.as_slice()?)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

pub fn bind_metrics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accuracy_score, m)?)?;
    m.add_function(wrap_pyfunction!(precision_score, m)?)?;
    m.add_function(wrap_pyfunction!(recall_score, m)?)?;
    m.add_function(wrap_pyfunction!(f1_score, m)?)?;
    m.add_function(wrap_pyfunction!(roc_auc_score, m)?)?;
    m.add_function(wrap_pyfunction!(mean_squared_error, m)?)?;
    m.add_function(wrap_pyfunction!(r2_score, m)?)?;
    Ok(())
}
