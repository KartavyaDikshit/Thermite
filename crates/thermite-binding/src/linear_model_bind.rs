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
    #[new]
    #[pyo3(signature = (fit_intercept=true))]
    fn new(fit_intercept: bool) -> Self {
        LinearRegression {
            core: CoreLinearRegression::new(fit_intercept),
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.core.fit(&X.as_array(), &y.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let preds = self.core.predict(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
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
    #[new]
    #[pyo3(signature = (alpha=1.0, fit_intercept=true))]
    fn new(alpha: f64, fit_intercept: bool) -> Self {
        Ridge {
            core: CoreRidge::new(alpha, fit_intercept),
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.core.fit(&X.as_array(), &y.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let preds = self.core.predict(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
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
    #[new]
    #[pyo3(signature = (alpha=1.0, fit_intercept=true, max_iter=1000, tol=1e-4))]
    fn new(alpha: f64, fit_intercept: bool, max_iter: usize, tol: f64) -> Self {
        Lasso {
            core: CoreLasso::new(alpha, fit_intercept, max_iter, tol),
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.core.fit(&X.as_array(), &y.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let preds = self.core.predict(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
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
    #[new]
    #[pyo3(signature = (C=1.0, max_iter=100, tol=1e-4, penalty="l2"))]
    fn new(C: f64, max_iter: usize, tol: f64, penalty: &str) -> Self {
        LogisticRegression {
            core: CoreLogisticRegression::new(C, max_iter, tol, penalty),
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.core.fit(&X.as_array(), &y.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)
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
        let data_slice = data.as_slice().unwrap();
        let indices_slice = indices.as_slice().unwrap();
        let indptr_slice = indptr.as_slice().unwrap();
        
        let cs_mat = build_csr(data_slice, indices_slice, indptr_slice, rows, cols)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            
        self.core.fit_sparse(&cs_mat, &y.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let preds = self.core.predict(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
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
        let data_slice = data.as_slice().unwrap();
        let indices_slice = indices.as_slice().unwrap();
        let indptr_slice = indptr.as_slice().unwrap();
        let cs_mat = build_csr(data_slice, indices_slice, indptr_slice, rows, cols)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let preds = self.core.predict_sparse(&cs_mat).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_array_bound(py, &preds))
    }

    fn predict_proba<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let preds = self.core.predict_proba(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
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
        let data_slice = data.as_slice().unwrap();
        let indices_slice = indices.as_slice().unwrap();
        let indptr_slice = indptr.as_slice().unwrap();
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

    fn fit(&mut self, X: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.core.fit(&X.as_array(), &y.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)
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
        let data_slice = data.as_slice().unwrap();
        let indices_slice = indices.as_slice().unwrap();
        let indptr_slice = indptr.as_slice().unwrap();
        
        let cs_mat = build_csr(data_slice, indices_slice, indptr_slice, rows, cols)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            
        self.core.fit_sparse(&cs_mat, &y.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let preds = self.core.predict(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
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
        let data_slice = data.as_slice().unwrap();
        let indices_slice = indices.as_slice().unwrap();
        let indptr_slice = indptr.as_slice().unwrap();
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
