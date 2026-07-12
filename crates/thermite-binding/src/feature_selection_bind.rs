
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::feature_selection::RFE as CoreRFE;

#[pyclass]
pub struct RFE {
    core: CoreRFE,
}

#[pymethods]
impl RFE {
    #[new]
    #[pyo3(signature = (n_features_to_select=5, step=1))]
    fn new(n_features_to_select: usize, step: usize) -> Self {
        RFE { core: CoreRFE::new(n_features_to_select, step) }
    }

    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_view = X.as_array(); let y_view = y.as_array(); py.allow_threads(|| self.core.fit(&x_view, &y_view).map_err(pyo3::exceptions::PyValueError::new_err))
    }

    fn transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        let res = py.allow_threads(|| self.core.transform(&x_view).map_err(pyo3::exceptions::PyValueError::new_err))?;
        Ok(PyArray2::from_array_bound(py, &res))
    }

    #[getter]
    fn support_(&self) -> Option<Vec<bool>> {
        self.core.support_.clone()
    }
    
    #[getter]
    fn ranking_(&self) -> Option<Vec<usize>> {
        self.core.ranking_.clone()
    }
}

pub fn bind_feature_selection(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RFE>()?;
    Ok(())
}
