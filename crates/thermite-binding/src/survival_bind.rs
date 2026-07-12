
use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::survival::SurvivalForest as CoreSurvivalForest;

#[pyclass]
pub struct SurvivalForest {
    core: CoreSurvivalForest,
}

#[pymethods]
impl SurvivalForest {
    #[new]
    #[pyo3(signature = (n_estimators=100))]
    fn new(n_estimators: usize) -> Self {
        SurvivalForest { core: CoreSurvivalForest::new(n_estimators) }
    }

    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, times: PyReadonlyArray1<f64>, events: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_view = X.as_array(); let times_view = times.as_array(); let events_view = events.as_array(); py.allow_threads(|| self.core.fit(&x_view, &times_view, &events_view).map_err(pyo3::exceptions::PyValueError::new_err))
    }

    fn predict_survival_function<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>, times_to_predict: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        let times_to_predict_view = times_to_predict.as_array();
        let res = py.allow_threads(|| self.core.predict_survival_function(&x_view, &times_to_predict_view).map_err(pyo3::exceptions::PyValueError::new_err))?;
        Ok(PyArray2::from_array_bound(py, &res))
    }
}

pub fn bind_survival(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SurvivalForest>()?;
    Ok(())
}
