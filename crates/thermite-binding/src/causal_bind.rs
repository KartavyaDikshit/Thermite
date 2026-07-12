use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::causal::TLearner as CoreTLearner;

#[pyclass]
pub struct TLearner {
    core: CoreTLearner,
}

#[pymethods]
impl TLearner {
    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = bincode::serialize(&self.core)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new_bound(py, &bytes))
    }

    fn __setstate__(&mut self, state: &Bound<'_, pyo3::types::PyBytes>) -> PyResult<()> {
        self.core = bincode::deserialize(state.as_bytes())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    #[new]
    fn new() -> Self {
        TLearner {
            core: CoreTLearner::new(),
        }
    }

    fn fit(&mut self, py: Python<'_>, x: PyReadonlyArray2<f64>, treatment: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_view = x.as_array();
        let t_view = treatment.as_array();
        let y_view = y.as_array();
        py.allow_threads(|| {
            self.core.fit(&x_view, &t_view, &y_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn predict_cate<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = x.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict_cate(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }
}

pub fn bind_causal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TLearner>()?;
    Ok(())
}
