use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::tree::{DecisionTreeClassifier as CoreDecisionTreeClassifier, DecisionTreeRegressor as CoreDecisionTreeRegressor};

#[pyclass]
pub struct DecisionTreeClassifier {
    core: CoreDecisionTreeClassifier,
}

#[pymethods]
impl DecisionTreeClassifier {
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
    #[pyo3(signature = (max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None))]
    fn new(
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        DecisionTreeClassifier {
            core: CoreDecisionTreeClassifier::new(
                max_depth,
                min_samples_split,
                min_samples_leaf,
                max_features,
                random_state,
            ),
        }
    }

    #[pyo3(signature = (X, y, categorical_features=None))]
    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>, categorical_features: Option<Vec<usize>>) -> PyResult<()> {
        if let Some(cf) = categorical_features {
            self.core.categorical_features = cf;
        } else {
            self.core.categorical_features = Vec::new();
        }
        let x_view = X.as_array();
        // Since y_slice is just a slice, we can borrow the array here instead of y_slice
        let y_view = y.as_array();
        let y_slice = y_view.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        py.allow_threads(|| {
            self.core.fit(&x_view, y_slice);
        });
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict(&x_view)
        });
        Ok(PyArray1::from_vec_bound(py, preds))
    }

    fn predict_proba<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        let probs = py.allow_threads(|| {
            self.core.predict_proba(&x_view)
        });
        Ok(PyArray2::from_array_bound(py, &probs))
    }
}

#[pyclass]
pub struct DecisionTreeRegressor {
    core: CoreDecisionTreeRegressor,
}

#[pymethods]
impl DecisionTreeRegressor {
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
    #[pyo3(signature = (max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None))]
    fn new(
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        DecisionTreeRegressor {
            core: CoreDecisionTreeRegressor::new(
                max_depth,
                min_samples_split,
                min_samples_leaf,
                max_features,
                random_state,
            ),
        }
    }

    #[pyo3(signature = (X, y, categorical_features=None))]
    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>, categorical_features: Option<Vec<usize>>) -> PyResult<()> {
        if let Some(cf) = categorical_features {
            self.core.categorical_features = cf;
        } else {
            self.core.categorical_features = Vec::new();
        }
        let x_view = X.as_array();
        let y_view = y.as_array();
        let y_slice = y_view.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        py.allow_threads(|| {
            self.core.fit(&x_view, y_slice);
        });
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict(&x_view)
        });
        Ok(PyArray1::from_vec_bound(py, preds))
    }
}

pub fn bind_tree(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DecisionTreeClassifier>()?;
    m.add_class::<DecisionTreeRegressor>()?;
    Ok(())
}
