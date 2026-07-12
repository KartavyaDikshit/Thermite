use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::neural_network::MLPClassifier as CoreMLPClassifier;

#[pyclass]
pub struct MLPClassifier {
    core: CoreMLPClassifier,
}

#[pymethods]
impl MLPClassifier {
    #[new]
    #[pyo3(signature = (hidden_layer_sizes=vec![100], learning_rate=0.001, max_iter=200, device="cpu"))]
    fn new(
        hidden_layer_sizes: Vec<usize>,
        learning_rate: f64,
        max_iter: usize,
        device: &str,
    ) -> Self {
        MLPClassifier {
            core: CoreMLPClassifier::new(hidden_layer_sizes, learning_rate, max_iter, device),
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

pub fn bind_neural_network(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MLPClassifier>()?;
    Ok(())
}
