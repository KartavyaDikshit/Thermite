use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::ensemble::{
    RandomForestClassifier as CoreRandomForestClassifier,
    RandomForestRegressor as CoreRandomForestRegressor,
    GradientBoostingClassifier as CoreGradientBoostingClassifier,
    GradientBoostingRegressor as CoreGradientBoostingRegressor,
    HistGradientBoostingClassifier as CoreHistGradientBoostingClassifier,
    HistGradientBoostingRegressor as CoreHistGradientBoostingRegressor,
    IsolationForest as CoreIsolationForest,
};
use thermite_gpu::DeviceKind;

#[pyclass]
pub struct HistGradientBoostingRegressor {
    core: CoreHistGradientBoostingRegressor,
}

#[pymethods]
impl HistGradientBoostingRegressor {
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

    fn save_checkpoint(&self, filepath: &str) -> PyResult<()> {
        let file = std::fs::File::create(filepath).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        bincode::serialize_into(file, &self.core).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    #[classmethod]
    fn load_checkpoint(_cls: &Bound<'_, pyo3::types::PyType>, filepath: &str) -> PyResult<Self> {
        let file = std::fs::File::open(filepath).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let core: CoreHistGradientBoostingRegressor = bincode::deserialize_from(file).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(HistGradientBoostingRegressor { core })
    }

    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=None, random_state=None))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        HistGradientBoostingRegressor {
            core: CoreHistGradientBoostingRegressor::new(n_estimators, learning_rate, max_depth, random_state),
        }
    }

    #[pyo3(signature = (X, y))]
    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_view = X.as_array();
        let y_view = y.as_array();
        py.allow_threads(|| {
            self.core.fit(&x_view, &y_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn predict<'py>(&mut self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }
}

#[pyclass]
pub struct HistGradientBoostingClassifier {
    core: CoreHistGradientBoostingClassifier,
}

#[pymethods]
impl HistGradientBoostingClassifier {
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

    fn save_checkpoint(&self, filepath: &str) -> PyResult<()> {
        let file = std::fs::File::create(filepath).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        bincode::serialize_into(file, &self.core).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    #[classmethod]
    fn load_checkpoint(_cls: &Bound<'_, pyo3::types::PyType>, filepath: &str) -> PyResult<Self> {
        let file = std::fs::File::open(filepath).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let core: CoreHistGradientBoostingClassifier = bincode::deserialize_from(file).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(HistGradientBoostingClassifier { core })
    }

    #[new]
    #[pyo3(signature = (n_estimators=100, learning_rate=0.1, max_depth=None, random_state=None))]
    fn new(
        n_estimators: usize,
        learning_rate: f64,
        max_depth: Option<usize>,
        random_state: Option<u64>,
    ) -> Self {
        HistGradientBoostingClassifier {
            core: CoreHistGradientBoostingClassifier::new(n_estimators, learning_rate, max_depth, random_state),
        }
    }

    #[pyo3(signature = (X, y))]
    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_view = X.as_array();
        let y_view = y.as_array();
        py.allow_threads(|| {
            self.core.fit(&x_view, &y_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn predict<'py>(&mut self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }

    fn predict_proba<'py>(&mut self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict_proba(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray2::from_array_bound(py, &preds))
    }
}

#[pyclass]
pub struct RandomForestClassifier {
    core: CoreRandomForestClassifier,
}

#[pymethods]
impl RandomForestClassifier {
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

    fn save_checkpoint(&self, filepath: &str) -> PyResult<()> {
        let file = std::fs::File::create(filepath).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        bincode::serialize_into(file, &self.core).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    #[classmethod]
    fn load_checkpoint(_cls: &Bound<'_, pyo3::types::PyType>, filepath: &str) -> PyResult<Self> {
        let file = std::fs::File::open(filepath).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let core: CoreRandomForestClassifier = bincode::deserialize_from(file).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(RandomForestClassifier { core })
    }
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
        core.device = DeviceKind::from_string(device);
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

    #[getter]
    fn estimators_(&self) -> PyResult<Vec<crate::tree_bind::PyTree>> {
        let mut res = Vec::with_capacity(self.core.estimators_.len());
        for est in &self.core.estimators_ {
            let n = est.nodes.len();
            let mut children_left = Vec::with_capacity(n);
            let mut children_right = Vec::with_capacity(n);
            let mut feature = Vec::with_capacity(n);
            let mut threshold = Vec::with_capacity(n);

            for node in &est.nodes {
                if node.is_leaf() {
                    children_left.push(-1);
                    children_right.push(-1);
                    feature.push(-2);
                    threshold.push(-2.0);
                } else {
                    children_left.push(node.left as isize);
                    children_right.push(node.right as isize);
                    feature.push(node.feature_idx as isize);
                    threshold.push(node.threshold);
                }
            }

            res.push(crate::tree_bind::PyTree {
                node_count: n,
                children_left,
                children_right,
                feature,
                threshold,
            });
        }
        Ok(res)
    }
}

#[pyclass]
pub struct RandomForestRegressor {
    core: CoreRandomForestRegressor,
}

#[pymethods]
impl RandomForestRegressor {
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

    fn save_checkpoint(&self, filepath: &str) -> PyResult<()> {
        let file = std::fs::File::create(filepath).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        bincode::serialize_into(file, &self.core).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    #[classmethod]
    fn load_checkpoint(_cls: &Bound<'_, pyo3::types::PyType>, filepath: &str) -> PyResult<Self> {
        let file = std::fs::File::open(filepath).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let core: CoreRandomForestRegressor = bincode::deserialize_from(file).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(RandomForestRegressor { core })
    }
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
        core.device = DeviceKind::from_string(device);
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

    #[getter]
    fn estimators_(&self) -> PyResult<Vec<crate::tree_bind::PyTree>> {
        let mut res = Vec::with_capacity(self.core.estimators_.len());
        for est in &self.core.estimators_ {
            let n = est.nodes.len();
            let mut children_left = Vec::with_capacity(n);
            let mut children_right = Vec::with_capacity(n);
            let mut feature = Vec::with_capacity(n);
            let mut threshold = Vec::with_capacity(n);

            for node in &est.nodes {
                if node.is_leaf() {
                    children_left.push(-1);
                    children_right.push(-1);
                    feature.push(-2);
                    threshold.push(-2.0);
                } else {
                    children_left.push(node.left as isize);
                    children_right.push(node.right as isize);
                    feature.push(node.feature_idx as isize);
                    threshold.push(node.threshold);
                }
            }

            res.push(crate::tree_bind::PyTree {
                node_count: n,
                children_left,
                children_right,
                feature,
                threshold,
            });
        }
        Ok(res)
    }
}

pub fn bind_ensemble(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RandomForestClassifier>()?;
    m.add_class::<RandomForestRegressor>()?;
    m.add_class::<GradientBoostingClassifier>()?;
    m.add_class::<GradientBoostingRegressor>()?;
    m.add_class::<HistGradientBoostingClassifier>()?;
    m.add_class::<HistGradientBoostingRegressor>()?;
    m.add_class::<IsolationForest>()?;
    Ok(())
}

#[pyclass]
pub struct IsolationForest {
    core: CoreIsolationForest,
}

#[pymethods]
impl IsolationForest {
    #[new]
    #[pyo3(signature = (n_estimators=100, random_state=None))]
    fn new(n_estimators: usize, random_state: Option<u64>) -> Self {
        IsolationForest {
            core: CoreIsolationForest::new(n_estimators, random_state),
        }
    }

    fn fit_predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let x_arr = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.fit_predict(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }
}

#[pyclass]
pub struct GradientBoostingRegressor {
    core: CoreGradientBoostingRegressor,
}

#[pymethods]
impl GradientBoostingRegressor {
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

    fn save_checkpoint(&self, filepath: &str) -> PyResult<()> {
        let file = std::fs::File::create(filepath).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        bincode::serialize_into(file, &self.core).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    #[classmethod]
    fn load_checkpoint(_cls: &Bound<'_, pyo3::types::PyType>, filepath: &str) -> PyResult<Self> {
        let file = std::fs::File::open(filepath).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let core: CoreGradientBoostingRegressor = bincode::deserialize_from(file).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(GradientBoostingRegressor { core })
    }
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

    fn save_checkpoint(&self, filepath: &str) -> PyResult<()> {
        let file = std::fs::File::create(filepath).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        bincode::serialize_into(file, &self.core).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    #[classmethod]
    fn load_checkpoint(_cls: &Bound<'_, pyo3::types::PyType>, filepath: &str) -> PyResult<Self> {
        let file = std::fs::File::open(filepath).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let core: CoreGradientBoostingClassifier = bincode::deserialize_from(file).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(GradientBoostingClassifier { core })
    }
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
