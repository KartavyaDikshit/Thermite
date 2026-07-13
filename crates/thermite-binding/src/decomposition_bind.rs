use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use thermite_core::decomposition::PCA as CorePCA;

#[pyclass]
pub struct PCA {
    core: CorePCA,
}

#[pymethods]
impl PCA {
    #[new]
    #[pyo3(signature = (n_components=2, random_state=None))]
    fn new(n_components: usize, random_state: Option<u64>) -> Self {
        PCA {
            core: CorePCA::new(n_components, random_state),
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>) -> PyResult<()> {
        self.core.fit(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let transformed = self.core.transform(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &transformed))
    }

    fn fit_transform<'py>(&mut self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let transformed = self.core.fit_transform(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &transformed))
    }

    fn inverse_transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let original = self.core.inverse_transform(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &original))
    }

    #[getter]
    fn components_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match &self.core.components_ {
            Some(c) => Ok(Some(PyArray2::from_array_bound(py, c))),
            None => Ok(None),
        }
    }

    #[getter]
    fn explained_variance_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.explained_variance_ {
            Some(v) => Ok(Some(PyArray1::from_array_bound(py, v))),
            None => Ok(None),
        }
    }

    #[getter]
    fn explained_variance_ratio_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.explained_variance_ratio_ {
            Some(v) => Ok(Some(PyArray1::from_array_bound(py, v))),
            None => Ok(None),
        }
    }

    #[getter]
    fn mean_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.mean_ {
            Some(m) => Ok(Some(PyArray1::from_array_bound(py, m))),
            None => Ok(None),
        }
    }
}

#[pyclass]
pub struct DictionaryLearning {
    core: thermite_core::decomposition::DictionaryLearning,
}

#[pymethods]
impl DictionaryLearning {
    #[new]
    #[pyo3(signature = (n_components=None, alpha=1.0, max_iter=1000, tol=1e-8, fit_algorithm="lars".to_string(), transform_algorithm="omp".to_string(), transform_alpha=None, n_jobs=None, random_state=None))]
    fn new(
        n_components: Option<usize>,
        alpha: f64,
        max_iter: usize,
        tol: f64,
        fit_algorithm: String,
        transform_algorithm: String,
        transform_alpha: Option<f64>,
        n_jobs: Option<i32>,
        random_state: Option<u64>,
    ) -> Self {
        DictionaryLearning {
            core: thermite_core::decomposition::DictionaryLearning::new(
                n_components, alpha, max_iter, tol, fit_algorithm, transform_algorithm, transform_alpha, n_jobs, random_state
            ),
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>) -> PyResult<()> {
        self.core.fit(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let transformed = self.core.transform(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &transformed))
    }

    #[getter]
    fn components_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match &self.core.components_ {
            Some(c) => Ok(Some(PyArray2::from_array_bound(py, c))),
            None => Ok(None),
        }
    }
}

pub fn bind_decomposition(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PCA>()?;
    m.add_class::<DictionaryLearning>()?;
    Ok(())
}
