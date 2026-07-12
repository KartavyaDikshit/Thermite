use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::automl::SurrogateOptimizer as CoreSurrogateOptimizer;

#[pyclass]
pub struct SurrogateOptimizer {
    core: CoreSurrogateOptimizer,
}

#[pymethods]
impl SurrogateOptimizer {
    #[new]
    #[pyo3(signature = (alpha=1.0))]
    fn new(alpha: f64) -> Self {
        SurrogateOptimizer {
            core: CoreSurrogateOptimizer::new(alpha),
        }
    }

    #[pyo3(signature = (X, y))]
    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_view = X.as_array();
        let y_view = y.as_array();
        let y_slice = y_view.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        
        // Wrap the slice into a 1D view
        let y_arr = ndarray::ArrayView1::from(y_slice);

        py.allow_threads(|| {
            self.core.fit(&x_view, &y_arr).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })?;
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict(&x_view).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })?;
        Ok(PyArray1::from_vec_bound(py, preds.into_raw_vec()))
    }

    fn suggest_next(&self, py: Python<'_>, X_candidates: PyReadonlyArray2<f64>) -> PyResult<usize> {
        let x_view = X_candidates.as_array();
        let best_idx = py.allow_threads(|| {
            self.core.suggest_next(&x_view).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })?;
        Ok(best_idx)
    }
}

pub fn bind_automl(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SurrogateOptimizer>()?;
    Ok(())
}
