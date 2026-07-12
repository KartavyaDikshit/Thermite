use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::ensemble::{
    RandomForestClassifier as CoreRandomForestClassifier,
    RandomForestRegressor as CoreRandomForestRegressor,
    GradientBoostingClassifier as CoreGradientBoostingClassifier,
    GradientBoostingRegressor as CoreGradientBoostingRegressor,
};
use thermite_gpu::DeviceKind;

#[pyclass]
pub struct RandomForestClassifier {
    core: CoreRandomForestClassifier,
}

#[pymethods]
impl RandomForestClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None, device="cpu"))]
    fn new(
        n_estimators: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        random_state: Option<u64>,
        device: &str,
    ) -> Self {
        let mut core = CoreRandomForestClassifier::new(
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            random_state,
        );
        core.device = DeviceKind::from_str(device);
        RandomForestClassifier { core }
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
        py.allow_threads(|| {
            self.core.fit(&x_view, &y_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }


}

#[pyclass]
pub struct RandomForestRegressor {
    core: CoreRandomForestRegressor,
}

#[pymethods]
impl RandomForestRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None, device="cpu"))]
    fn new(
        n_estimators: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        max_features: Option<usize>,
        random_state: Option<u64>,
        device: &str,
    ) -> Self {
        let mut core = CoreRandomForestRegressor::new(
            n_estimators,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features,
            random_state,
        );
        core.device = DeviceKind::from_str(device);
        RandomForestRegressor { core }
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
        py.allow_threads(|| {
            self.core.fit(&x_view, &y_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }


}

pub fn bind_ensemble(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RandomForestClassifier>()?;
    m.add_class::<RandomForestRegressor>()?;
    m.add_class::<GradientBoostingClassifier>()?;
    m.add_class::<GradientBoostingRegressor>()?;
    Ok(())
}

#[pyclass]
pub struct GradientBoostingRegressor {
    core: CoreGradientBoostingRegressor,
}

#[pymethods]
impl GradientBoostingRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=None, random_state=None))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        GradientBoostingRegressor {
            core: CoreGradientBoostingRegressor::new(n_estimators, learning_rate, max_depth, random_state),
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
        py.allow_threads(|| {
            self.core.fit(&x_view, &y_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }
}

#[pyclass]
pub struct GradientBoostingClassifier {
    core: CoreGradientBoostingClassifier,
}

#[pymethods]
impl GradientBoostingClassifier {
    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=None, random_state=None))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        GradientBoostingClassifier {
            core: CoreGradientBoostingClassifier::new(n_estimators, learning_rate, max_depth, random_state),
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
        py.allow_threads(|| {
            self.core.fit(&x_view, &y_view).map_err(pyo3::exceptions::PyValueError::new_err)
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
        let preds = py.allow_threads(|| {
            self.core.predict_proba(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray2::from_array_bound(py, &preds))
    }
}
