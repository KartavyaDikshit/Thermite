use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::tree::{DecisionTreeClassifier as CoreDecisionTreeClassifier, DecisionTreeRegressor as CoreDecisionTreeRegressor};

#[pyclass]
pub struct DecisionTreeClassifier {
    core: CoreDecisionTreeClassifier,
}

#[pymethods]
impl DecisionTreeClassifier {
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
    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>, categorical_features: Option<Vec<usize>>) -> PyResult<()> {
        let y_slice = match y.as_slice() {
            Ok(s) => s,
            Err(_) => return Err(pyo3::exceptions::PyValueError::new_err("y must be contiguous")),
        };
        if let Some(cf) = categorical_features {
            self.core.categorical_features = cf;
        } else {
            self.core.categorical_features = Vec::new();
        }
        self.core.fit(&X.as_array(), y_slice);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let preds = self.core.predict(&X.as_array());
        Ok(PyArray1::from_vec_bound(py, preds))
    }

    fn predict_proba<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let probs = self.core.predict_proba(&X.as_array());
        Ok(PyArray2::from_array_bound(py, &probs))
    }
}

#[pyclass]
pub struct DecisionTreeRegressor {
    core: CoreDecisionTreeRegressor,
}

#[pymethods]
impl DecisionTreeRegressor {
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
    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>, categorical_features: Option<Vec<usize>>) -> PyResult<()> {
        let y_slice = match y.as_slice() {
            Ok(s) => s,
            Err(_) => return Err(pyo3::exceptions::PyValueError::new_err("y must be contiguous")),
        };
        if let Some(cf) = categorical_features {
            self.core.categorical_features = cf;
        } else {
            self.core.categorical_features = Vec::new();
        }
        self.core.fit(&X.as_array(), y_slice);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let preds = self.core.predict(&X.as_array());
        Ok(PyArray1::from_vec_bound(py, preds))
    }
}

pub fn bind_tree(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DecisionTreeClassifier>()?;
    m.add_class::<DecisionTreeRegressor>()?;
    Ok(())
}
