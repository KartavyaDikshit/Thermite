use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use thermite_core::cross_decomposition::{PLSRegression as CorePLSRegression, CCA as CoreCCA};

#[pyclass]
pub struct PLSRegression {
    core: CorePLSRegression,
}

#[pymethods]
impl PLSRegression {
    #[new]
    #[pyo3(signature = (n_components=2, scale=true, max_iter=500, tol=1e-06, copy=true))]
    fn new(n_components: usize, scale: bool, max_iter: usize, tol: f64, copy: bool) -> Self {
        PLSRegression {
            core: CorePLSRegression::new(n_components, scale, max_iter, tol, copy),
        }
    }

    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, Y: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_view = X.as_array();
        let y_view = Y.as_array();
        py.allow_threads(|| {
            self.core.fit(&x_view, &y_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray2::from_array_bound(py, &preds))
    }
}

#[pyclass]
pub struct CCA {
    core: CoreCCA,
}

#[pymethods]
impl CCA {
    #[new]
    #[pyo3(signature = (n_components=2, scale=true, max_iter=500, tol=1e-06, copy=true))]
    fn new(n_components: usize, scale: bool, max_iter: usize, tol: f64, copy: bool) -> Self {
        CCA {
            core: CoreCCA::new(n_components, scale, max_iter, tol, copy),
        }
    }

    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, Y: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_view = X.as_array();
        let y_view = Y.as_array();
        py.allow_threads(|| {
            self.core.fit(&x_view, &y_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray2::from_array_bound(py, &preds))
    }
}

pub fn bind_cross_decomposition(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PLSRegression>()?;
    m.add_class::<CCA>()?;
    Ok(())
}
