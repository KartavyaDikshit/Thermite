use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::naive_bayes::GaussianNB as CoreGaussianNB;

#[pyclass]
pub struct GaussianNB {
    core: CoreGaussianNB,
}

#[pymethods]
impl GaussianNB {
    #[new]
    fn new() -> Self {
        GaussianNB {
            core: CoreGaussianNB::new(),
        }
    }

    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_view = X.as_array();
        let y_view = y.as_array();
        py.allow_threads(|| {
            self.core.fit(&x_view, &y_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    #[pyo3(signature = (X, y, classes=None))]
    fn partial_fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>, classes: Option<Vec<f64>>) -> PyResult<()> {
        let x_view = X.as_array();
        let y_view = y.as_array();
        py.allow_threads(|| {
            self.core.partial_fit(&x_view, &y_view, classes).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }

    fn predict_proba<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        let proba = py.allow_threads(|| {
            self.core.predict_proba(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray2::from_array_bound(py, &proba))
    }
}

pub fn bind_naive_bayes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GaussianNB>()?;
    Ok(())
}
