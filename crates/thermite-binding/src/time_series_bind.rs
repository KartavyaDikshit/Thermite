
use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use thermite_core::time_series::AutoRegressive as CoreAutoRegressive;

#[pyclass]
pub struct AutoRegressive {
    core: CoreAutoRegressive,
}

#[pymethods]
impl AutoRegressive {
    #[new]
    #[pyo3(signature = (lags=1))]
    fn new(lags: usize) -> Self {
        AutoRegressive { core: CoreAutoRegressive::new(lags) }
    }

    fn fit(&mut self, py: Python<'_>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let y_view = y.as_array(); py.allow_threads(|| self.core.fit(&y_view).map_err(pyo3::exceptions::PyValueError::new_err))
    }

    fn predict<'py>(&self, py: Python<'py>, steps: usize, last_y: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let last_y_view = last_y.as_array();
        let res = py.allow_threads(|| self.core.predict(steps, &last_y_view).map_err(pyo3::exceptions::PyValueError::new_err))?;
        Ok(PyArray1::from_array_bound(py, &res))
    }
}

pub fn bind_time_series(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AutoRegressive>()?;
    Ok(())
}
