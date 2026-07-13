
use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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

#[pyclass]
pub struct SequentialFeatureSelector {
    core: thermite_core::feature_selection::SequentialFeatureSelector,
}

#[pymethods]
impl SequentialFeatureSelector {
    #[new]
    #[pyo3(signature = (n_features_to_select=None, tol=None, direction="forward".to_string(), cv=5, n_jobs=None))]
    fn new(
        n_features_to_select: Option<usize>,
        tol: Option<f64>,
        direction: String,
        cv: usize,
        n_jobs: Option<i32>
    ) -> Self {
        SequentialFeatureSelector {
            core: thermite_core::feature_selection::SequentialFeatureSelector::new(
                n_features_to_select, tol, direction, cv, n_jobs
            ),
        }
    }

    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_view = X.as_array();
        let y_view = y.as_array();
        py.allow_threads(|| self.core.fit(&x_view, &y_view).map_err(pyo3::exceptions::PyValueError::new_err))
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
}

pub fn bind_feature_selection(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RFE>()?;
    m.add_class::<SequentialFeatureSelector>()?;
    Ok(())
}
