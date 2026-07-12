use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::linear_model::{LinearRegression as CoreLinearRegression, Ridge as CoreRidge, Lasso as CoreLasso, LogisticRegression as CoreLogisticRegression, LinearSVC as CoreLinearSVC};
use thermite_core::sparse::build_csr;

#[pyclass]
pub struct LinearRegression {
    core: CoreLinearRegression,
}

#[pymethods]
impl LinearRegression {
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
    #[pyo3(signature = (fit_intercept=true))]
    fn new(fit_intercept: bool) -> Self {
        LinearRegression {
            core: CoreLinearRegression::new(fit_intercept),
        }
    }

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

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.coef_ {
            Some(c) => Ok(Some(PyArray1::from_array_bound(py, c))),
            None => Ok(None),
        }
    }

    #[getter]
    fn intercept_(&self) -> f64 {
        self.core.intercept_
    }
}

#[pyclass]
pub struct Ridge {
    core: CoreRidge,
}

#[pymethods]
impl Ridge {
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
    #[pyo3(signature = (alpha=1.0, fit_intercept=true))]
    fn new(alpha: f64, fit_intercept: bool) -> Self {
        Ridge {
            core: CoreRidge::new(alpha, fit_intercept),
        }
    }

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

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.coef_ {
            Some(c) => Ok(Some(PyArray1::from_array_bound(py, c))),
            None => Ok(None),
        }
    }

    #[getter]
    fn intercept_(&self) -> f64 {
        self.core.intercept_
    }
}

#[pyclass]
pub struct Lasso {
    core: CoreLasso,
}

#[pymethods]
impl Lasso {
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
    #[pyo3(signature = (alpha=1.0, fit_intercept=true, max_iter=1000, tol=1e-4))]
    fn new(alpha: f64, fit_intercept: bool, max_iter: usize, tol: f64) -> Self {
        Lasso {
            core: CoreLasso::new(alpha, fit_intercept, max_iter, tol),
        }
    }

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

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.coef_ {
            Some(c) => Ok(Some(PyArray1::from_array_bound(py, c))),
            None => Ok(None),
        }
    }

    #[getter]
    fn intercept_(&self) -> f64 {
        self.core.intercept_
    }
}

#[pyclass]
pub struct LogisticRegression {
    core: CoreLogisticRegression,
}

#[pymethods]
impl LogisticRegression {
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
    #[pyo3(signature = (C=1.0, max_iter=100, tol=1e-4, penalty="l2"))]
    fn new(C: f64, max_iter: usize, tol: f64, penalty: &str) -> Self {
        LogisticRegression {
            core: CoreLogisticRegression::new(C, max_iter, tol, penalty),
        }
    }

    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_view = X.as_array();
        let y_view = y.as_array();
        py.allow_threads(|| {
            self.core.fit(&x_view, &y_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    #[pyo3(signature = (X, y, classes=None))]
    fn partial_fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>, classes: Option<Vec<f64>>) -> PyResult<()> {
        let x_view = X.as_array();
        let y_view = y.as_array();
        py.allow_threads(|| {
            self.core.partial_fit(&x_view, &y_view, classes).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    #[pyo3(signature = (data, indices, indptr, rows, cols, y))]
    fn fit_sparse(
        &mut self,
        data: PyReadonlyArray1<f64>,
        indices: PyReadonlyArray1<usize>,
        indptr: PyReadonlyArray1<usize>,
        rows: usize,
        cols: usize,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let data_slice = data.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        let indices_slice = indices.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        let indptr_slice = indptr.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        
        let cs_mat = build_csr(data_slice, indices_slice, indptr_slice, rows, cols)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            
        self.core.fit_sparse(&cs_mat, &y.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }

    #[pyo3(signature = (data, indices, indptr, rows, cols))]
    fn predict_sparse<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
        indices: PyReadonlyArray1<usize>,
        indptr: PyReadonlyArray1<usize>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data_slice = data.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        let indices_slice = indices.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        let indptr_slice = indptr.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        let cs_mat = build_csr(data_slice, indices_slice, indptr_slice, rows, cols)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let preds = self.core.predict_sparse(&cs_mat).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }

    fn predict_proba<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict_proba(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray2::from_array_bound(py, &preds))
    }

    #[pyo3(signature = (data, indices, indptr, rows, cols))]
    fn predict_proba_sparse<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
        indices: PyReadonlyArray1<usize>,
        indptr: PyReadonlyArray1<usize>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let data_slice = data.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        let indices_slice = indices.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        let indptr_slice = indptr.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        let cs_mat = build_csr(data_slice, indices_slice, indptr_slice, rows, cols)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let preds = self.core.predict_proba_sparse(&cs_mat).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &preds))
    }

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match &self.core.coef_ {
            Some(c) => Ok(Some(PyArray2::from_array_bound(py, c))),
            None => Ok(None),
        }
    }

    #[getter]
    fn intercept_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.intercept_ {
            Some(c) => Ok(Some(PyArray1::from_array_bound(py, c))),
            None => Ok(None),
        }
    }
}

pub fn bind_linear_model(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LinearRegression>()?;
    m.add_class::<Ridge>()?;
    m.add_class::<Lasso>()?;
    m.add_class::<LogisticRegression>()?;
    m.add_class::<LinearSVC>()?;
    Ok(())
}

#[pyclass]
pub struct LinearSVC {
    core: CoreLinearSVC,
}

#[pymethods]
impl LinearSVC {
    #[new]
    #[pyo3(signature = (C=1.0, max_iter=1000, tol=1e-4))]
    fn new(C: f64, max_iter: usize, tol: f64) -> Self {
        LinearSVC {
            core: CoreLinearSVC::new(C, max_iter, tol),
        }
    }

    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_view = X.as_array();
        let y_view = y.as_array();
        py.allow_threads(|| {
            self.core.fit(&x_view, &y_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    #[pyo3(signature = (data, indices, indptr, rows, cols, y))]
    fn fit_sparse(
        &mut self,
        data: PyReadonlyArray1<f64>,
        indices: PyReadonlyArray1<usize>,
        indptr: PyReadonlyArray1<usize>,
        rows: usize,
        cols: usize,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let data_slice = data.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        let indices_slice = indices.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        let indptr_slice = indptr.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        
        let cs_mat = build_csr(data_slice, indices_slice, indptr_slice, rows, cols)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            
        self.core.fit_sparse(&cs_mat, &y.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_view = X.as_array();
        let preds = py.allow_threads(|| {
            self.core.predict(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }

    #[pyo3(signature = (data, indices, indptr, rows, cols))]
    fn predict_sparse<'py>(
        &self,
        py: Python<'py>,
        data: PyReadonlyArray1<f64>,
        indices: PyReadonlyArray1<usize>,
        indptr: PyReadonlyArray1<usize>,
        rows: usize,
        cols: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let data_slice = data.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        let indices_slice = indices.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        let indptr_slice = indptr.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("Array must be contiguous"))?;
        let cs_mat = build_csr(data_slice, indices_slice, indptr_slice, rows, cols)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let preds = self.core.predict_sparse(&cs_mat).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }

    #[getter]
    fn coef_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match &self.core.coef_ {
            Some(c) => Ok(Some(PyArray2::from_array_bound(py, c))),
            None => Ok(None),
        }
    }

    #[getter]
    fn intercept_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.intercept_ {
            Some(c) => Ok(Some(PyArray1::from_array_bound(py, c))),
            None => Ok(None),
        }
    }

    #[getter]
    fn classes_(&self) -> Option<Vec<f64>> {
        self.core.classes_.clone()
    }
}
