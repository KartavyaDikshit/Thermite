use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::svm::SVC as CoreSVC;

#[pyclass]
pub struct SVC {
    core: CoreSVC,
}

#[pymethods]
impl SVC {
    #[new]
    #[pyo3(signature = (C=1.0, kernel="rbf", degree=3, gamma="scale", coef0=0.0, probability=false, eps=1e-3, max_iter=1000))]
    fn new(
        C: f64,
        kernel: &str,
        degree: i32,
        gamma: &str,
        coef0: f64,
        probability: bool,
        eps: f64,
        max_iter: i32,
    ) -> Self {
        SVC {
            core: CoreSVC::new(
                C,
                kernel.to_string(),
                degree,
                gamma.to_string(),
                coef0,
                probability,
                eps,
                max_iter,
            ),
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_arr = X.as_array();
        let y_arr = y.as_array();
        self.core.fit(&x_arr, &y_arr).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_arr = X.as_array();
        let preds = self.core.predict(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }

    fn predict_proba<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        let probs = self.core.predict_proba(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &probs))
    }

    #[getter]
    fn classes_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.classes_ {
            Some(classes) => Ok(Some(PyArray1::from_vec_bound(py, classes.clone()))),
            None => Ok(None),
        }
    }
}

pub fn bind_svm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SVC>()?;
    Ok(())
}
