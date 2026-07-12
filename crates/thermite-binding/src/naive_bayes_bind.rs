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

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let y_slice = match y.as_slice() {
            Ok(s) => s,
            Err(_) => return Err(pyo3::exceptions::PyValueError::new_err("y must be contiguous")),
        };
        self.core.fit(&X.as_array(), &y.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let preds = self.core.predict(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }

    fn predict_proba<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let proba = self.core.predict_proba(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &proba))
    }
}

pub fn bind_naive_bayes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GaussianNB>()?;
    Ok(())
}
