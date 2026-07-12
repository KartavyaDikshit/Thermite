use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::neighbors::{KNeighborsClassifier as CoreKNeighborsClassifier, WeightType};

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
    Ok(())
}
