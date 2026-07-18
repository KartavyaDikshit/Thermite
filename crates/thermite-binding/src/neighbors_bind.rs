use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::neighbors::{Algorithm, KNeighborsClassifier as CoreKNeighborsClassifier, KNeighborsRegressor as CoreKNeighborsRegressor, LocalOutlierFactor as CoreLocalOutlierFactor, WeightType};

#[pyclass]
pub struct KNeighborsClassifier {
    core: CoreKNeighborsClassifier,
}

#[pymethods]
impl KNeighborsClassifier {
    #[new]
    #[pyo3(signature = (n_neighbors=5, weights="uniform", algorithm="brute"))]
    fn new(n_neighbors: usize, weights: &str, algorithm: &str) -> PyResult<Self> {
        let weight_type = WeightType::from_str(weights).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let alg = Algorithm::from_str(algorithm).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(KNeighborsClassifier {
            core: CoreKNeighborsClassifier::new(n_neighbors, weight_type, alg),
        })
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<i64>) -> PyResult<()> {
        let y_arr = y.as_array();
        self.core.fit(&X.as_array(), &y_arr).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let preds = self.core.predict(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }

    fn predict_proba<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let probs = self.core.predict_proba(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &probs))
    }

    #[pyo3(signature = (X, n_neighbors=None, return_distance=true))]
    fn kneighbors<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
        n_neighbors: Option<usize>,
        return_distance: bool,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let (dists, indices) = self.core.kneighbors(&X.as_array(), n_neighbors)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let dists_py = PyArray2::from_array_bound(py, &dists);
        let indices_py = PyArray2::from_array_bound(py, &indices.mapv(|i| i as f64));
        Ok((dists_py, indices_py))
    }

    #[pyo3(signature = (X, radius))]
    fn radius_neighbors<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
        radius: f64,
    ) -> PyResult<PyObject> {
        let (dists, indices) = self.core.radius_neighbors(&X.as_array(), radius)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let dists_py: Vec<Bound<'py, PyArray1<f64>>> = dists.iter()
            .map(|d| PyArray1::from_array_bound(py, d))
            .collect();
        let indices_py: Vec<Bound<'py, PyArray1<f64>>> = indices.iter()
            .map(|i| PyArray1::from_array_bound(py, &i.mapv(|v| v as f64)))
            .collect();
        let result = (dists_py, indices_py);
        Ok(result.into_py(py))
    }
}

pub fn bind_neighbors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KNeighborsClassifier>()?;
    m.add_class::<KNeighborsRegressor>()?;
    m.add_class::<LocalOutlierFactor>()?;
    Ok(())
}

#[pyclass]
pub struct KNeighborsRegressor {
    core: CoreKNeighborsRegressor,
}

#[pymethods]
impl KNeighborsRegressor {
    #[new]
    #[pyo3(signature = (n_neighbors=5, weights="uniform", algorithm="brute"))]
    fn new(n_neighbors: usize, weights: &str, algorithm: &str) -> PyResult<Self> {
        let weight_type = WeightType::from_str(weights).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let alg = Algorithm::from_str(algorithm).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(KNeighborsRegressor {
            core: CoreKNeighborsRegressor::new(n_neighbors, weight_type, alg),
        })
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let y_arr = y.as_array();
        self.core.fit(&X.as_array(), &y_arr).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let preds = self.core.predict(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }

    #[pyo3(signature = (X, n_neighbors=None, return_distance=true))]
    fn kneighbors<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
        n_neighbors: Option<usize>,
        return_distance: bool,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let (dists, indices) = self.core.kneighbors(&X.as_array(), n_neighbors)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let dists_py = PyArray2::from_array_bound(py, &dists);
        let indices_py = PyArray2::from_array_bound(py, &indices.mapv(|i| i as f64));
        Ok((dists_py, indices_py))
    }

    #[pyo3(signature = (X, radius))]
    fn radius_neighbors<'py>(
        &self,
        py: Python<'py>,
        X: PyReadonlyArray2<f64>,
        radius: f64,
    ) -> PyResult<PyObject> {
        let (dists, indices) = self.core.radius_neighbors(&X.as_array(), radius)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let dists_py: Vec<Bound<'py, PyArray1<f64>>> = dists.iter()
            .map(|d| PyArray1::from_array_bound(py, d))
            .collect();
        let indices_py: Vec<Bound<'py, PyArray1<f64>>> = indices.iter()
            .map(|i| PyArray1::from_array_bound(py, &i.mapv(|v| v as f64)))
            .collect();
        let result = (dists_py, indices_py);
        Ok(result.into_py(py))
    }
}

#[pyclass]
pub struct LocalOutlierFactor {
    core: CoreLocalOutlierFactor,
}

#[pymethods]
impl LocalOutlierFactor {
    #[new]
    #[pyo3(signature = (n_neighbors=20, contamination=0.1))]
    fn new(n_neighbors: usize, contamination: f64) -> Self {
        LocalOutlierFactor {
            core: CoreLocalOutlierFactor::new(n_neighbors, contamination),
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
