use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use thermite_core::mixture::GaussianMixture as CoreGaussianMixture;

#[pyclass]
pub struct GaussianMixture {
    core: CoreGaussianMixture,
}

#[pymethods]
impl GaussianMixture {
    #[new]
    #[pyo3(signature = (n_components=1, max_iter=100, tol=1e-3))]
    fn new(n_components: usize, max_iter: usize, tol: f64) -> Self {
        GaussianMixture {
            core: CoreGaussianMixture::new(n_components, max_iter, tol),
        }
    }

    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_view = X.as_array();
        py.allow_threads(|| {
            self.core.fit(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<usize>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray1::from_vec_bound(py, preds))
    }

    fn predict_proba<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict_proba(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray2::from_array_bound(py, &preds))
    }
}

pub fn bind_mixture(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GaussianMixture>()?;
    Ok(())
}
