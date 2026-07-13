use pyo3::prelude::*;
use numpy::PyReadonlyArray2;
use thermite_core::recommender::ALS as CoreALS;

#[pyclass]
pub struct ALS {
    core: CoreALS,
}

#[pymethods]
impl ALS {
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
    #[pyo3(signature = (factors=10, iterations=10, regularization=0.1))]
    fn new(factors: usize, iterations: usize, regularization: f64) -> Self {
        ALS {
            core: CoreALS::new(factors, iterations, regularization),
        }
    }

    fn fit(&mut self, py: Python<'_>, R: PyReadonlyArray2<f64>) -> PyResult<()> {
        let r_view = R.as_array();
        py.allow_threads(|| {
            self.core.fit(&r_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn predict(&self, _py: Python<'_>, user_id: usize, item_id: usize) -> PyResult<f64> {
        self.core.predict(user_id, item_id).map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

pub fn bind_recommender(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ALS>()?;
    Ok(())
}

