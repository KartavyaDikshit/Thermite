use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, IntoPyArray};
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

#[pyfunction]
pub fn log_loss(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>) -> PyResult<f64> {
    core_metrics::log_loss(y_true.as_slice()?, y_pred.as_slice()?)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
pub fn mean_absolute_percentage_error(y_true: PyReadonlyArray1<f64>, y_pred: PyReadonlyArray1<f64>) -> PyResult<f64> {
    core_metrics::mean_absolute_percentage_error(y_true.as_slice()?, y_pred.as_slice()?)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

#[pyfunction]
pub fn pairwise_distances<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    metric: &str,
) -> PyResult<Bound<'py, pyo3::PyAny>> {
    let x_shape = x.shape();
    let y_shape = y.shape();
    if x_shape[1] != y_shape[1] {
        return Err(pyo3::exceptions::PyValueError::new_err("Number of features must match"));
    }
    
    let x_slice = x.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("X is not contiguous"))?;
    let y_slice = y.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Y is not contiguous"))?;
    
    let distances = core_metrics::pairwise_distances(
        x_slice,
        x_shape[0],
        y_slice,
        y_shape[0],
        x_shape[1],
        metric,
    ).map_err(pyo3::exceptions::PyValueError::new_err)?;
    
    use numpy::ndarray::Array2;
    let dist_array = Array2::from_shape_vec((x_shape[0], y_shape[0]), distances)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    Ok(dist_array.into_pyarray_bound(py).into_any())
}

pub fn bind_metrics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accuracy_score, m)?)?;
    m.add_function(wrap_pyfunction!(precision_score, m)?)?;
    m.add_function(wrap_pyfunction!(recall_score, m)?)?;
    m.add_function(wrap_pyfunction!(f1_score, m)?)?;
    m.add_function(wrap_pyfunction!(roc_auc_score, m)?)?;
    m.add_function(wrap_pyfunction!(mean_squared_error, m)?)?;
    m.add_function(wrap_pyfunction!(r2_score, m)?)?;
    m.add_function(wrap_pyfunction!(log_loss, m)?)?;
    m.add_function(wrap_pyfunction!(mean_absolute_percentage_error, m)?)?;
    m.add_function(wrap_pyfunction!(pairwise_distances, m)?)?;
    Ok(())
}
