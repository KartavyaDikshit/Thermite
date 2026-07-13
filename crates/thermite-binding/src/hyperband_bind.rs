use pyo3::prelude::*;
use pyo3::types::PyDict;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use std::collections::HashMap;
use thermite_core::hyperband::{IncrementalEstimator, SuccessiveHalvingSearchCV as CoreSuccessiveHalvingSearchCV};

pub struct PyIncrementalEstimator {
    pub obj: PyObject,
}

impl Clone for PyIncrementalEstimator {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            let copy_mod = PyModule::import_bound(py, "copy").expect("Failed to import copy");
            let cloned_obj = copy_mod.call_method1("deepcopy", (&self.obj,)).expect("Failed to deepcopy estimator").into();
            PyIncrementalEstimator { obj: cloned_obj }
        })
    }
}

impl IncrementalEstimator for PyIncrementalEstimator {
    fn set_params(&mut self, params: &HashMap<String, f64>) {
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            for (k, v) in params {
                dict.set_item(k, v).unwrap();
            }
            let kwargs = PyDict::new_bound(py);
            kwargs.update(dict.as_mapping()).unwrap();
            self.obj.call_method_bound(py, "set_params", (), Some(&kwargs)).unwrap();
        });
    }

    fn partial_fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), String> {
        Python::with_gil(|py| {
            let x_py = x.to_owned().into_pyarray_bound(py);
            let y_py = y.to_owned().into_pyarray_bound(py);
            self.obj.call_method1(py, "partial_fit", (x_py, y_py)).map_err(|e| format!("Python error: {}", e))?;
            Ok(())
        })
    }

    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64, String> {
        Python::with_gil(|py| {
            let x_py = x.to_owned().into_pyarray_bound(py);
            let y_py = y.to_owned().into_pyarray_bound(py);
            let res = self.obj.call_method1(py, "score", (x_py, y_py)).map_err(|e| format!("Python error: {}", e))?;
            let score: f64 = res.extract(py).map_err(|e| format!("Extraction error: {}", e))?;
            Ok(score)
        })
    }
}

#[pyclass]
pub struct SuccessiveHalvingSearchCV {
    core: CoreSuccessiveHalvingSearchCV<PyIncrementalEstimator>,
}

#[pymethods]
impl SuccessiveHalvingSearchCV {
    #[new]
    #[pyo3(signature = (estimator, param_grid, min_resources=10, factor=3))]
    fn new(_py: Python<'_>, estimator: PyObject, param_grid: Vec<Bound<'_, PyDict>>, min_resources: usize, factor: usize) -> PyResult<Self> {
        let mut rust_param_grid = Vec::new();
        for dict in param_grid {
            let mut params = HashMap::new();
            for (k, v) in dict {
                let key: String = k.extract()?;
                let val: f64 = v.extract()?;
                params.insert(key, val);
            }
            rust_param_grid.push(params);
        }

        let base_estimator = PyIncrementalEstimator { obj: estimator };
        Ok(SuccessiveHalvingSearchCV {
            core: CoreSuccessiveHalvingSearchCV::new(base_estimator, rust_param_grid, min_resources, factor),
        })
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_arr = X.as_array().to_owned();
        let y_arr = y.as_array().to_owned();
        self.core.fit(&x_arr, &y_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(())
    }

    #[getter]
    fn best_estimator_<'py>(&self, py: Python<'py>) -> PyResult<Option<PyObject>> {
        match &self.core.best_estimator_ {
            Some(est) => Ok(Some(est.obj.clone_ref(py))),
            None => Ok(None),
        }
    }

    #[getter]
    fn best_score_(&self) -> f64 {
        self.core.best_score_
    }
}

pub fn bind_hyperband(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SuccessiveHalvingSearchCV>()?;
    Ok(())
}
