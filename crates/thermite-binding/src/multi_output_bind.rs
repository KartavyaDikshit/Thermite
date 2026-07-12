use pyo3::prelude::*;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyArrayMethods};
use thermite_core::multi_output::{Estimator, MultiOutputRegressor as CoreMultiOutputRegressor};

pub struct PyEstimator {
    pub obj: PyObject,
}

impl Clone for PyEstimator {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            let copy_mod = PyModule::import_bound(py, "copy").expect("Failed to import copy");
            let cloned_obj = copy_mod.call_method1("deepcopy", (&self.obj,)).expect("Failed to deepcopy estimator").into();
            PyEstimator { obj: cloned_obj }
        })
    }
}

impl Estimator for PyEstimator {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        Python::with_gil(|py| {
            let x_py = x.to_owned().into_pyarray_bound(py);
            let y_py = y.to_owned().into_pyarray_bound(py);
            self.obj.call_method1(py, "fit", (x_py, y_py)).map_err(|e| format!("Python error: {}", e))?;
            Ok(())
        })
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, String> {
        Python::with_gil(|py| {
            let x_py = x.to_owned().into_pyarray_bound(py);
            let preds_obj = self.obj.call_method1(py, "predict", (x_py,)).map_err(|e| format!("Python error: {}", e))?;
            let preds: Bound<'_, PyArray1<f64>> = preds_obj.extract(py).map_err(|e| format!("Failed to extract Array1: {}", e))?;
            Ok(preds.to_owned_array())
        })
    }
}

#[pyclass]
pub struct MultiOutputRegressor {
    core: CoreMultiOutputRegressor<PyEstimator>,
}

#[pymethods]
impl MultiOutputRegressor {
    #[new]
    fn new(estimator: PyObject) -> Self {
        MultiOutputRegressor {
            core: CoreMultiOutputRegressor::new(PyEstimator { obj: estimator }),
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_arr = X.as_array().to_owned();
        let y_arr = y.as_array().to_owned();
        self.core.fit(&x_arr, &y_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array().to_owned();
        let preds = self.core.predict(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(preds.into_pyarray_bound(py))
    }

    #[getter]
    fn estimators<'py>(&self, py: Python<'py>) -> PyResult<Vec<PyObject>> {
        let mut list = Vec::new();
        for est in &self.core.estimators {
            list.push(est.obj.clone_ref(py));
        }
        Ok(list)
    }
}

pub fn bind_multi_output(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MultiOutputRegressor>()?;
    Ok(())
}
