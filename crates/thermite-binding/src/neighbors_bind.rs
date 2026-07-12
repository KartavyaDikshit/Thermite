use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::neighbors::{KNeighborsClassifier as CoreKNeighborsClassifier, LocalOutlierFactor as CoreLocalOutlierFactor, WeightType};

#[pyclass]
pub struct KNeighborsClassifier {
    core: CoreKNeighborsClassifier,
}

#[pymethods]
impl KNeighborsClassifier {
    #[new]
    #[pyo3(signature = (n_neighbors=5, weights="uniform"))]
    fn new(n_neighbors: usize, weights: &str) -> PyResult<Self> {
        let weight_type = WeightType::from_str(weights).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(KNeighborsClassifier {
            core: CoreKNeighborsClassifier::new(n_neighbors, weight_type),
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
}

pub fn bind_neighbors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KNeighborsClassifier>()?;
    m.add_class::<LocalOutlierFactor>()?;
    Ok(())
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
