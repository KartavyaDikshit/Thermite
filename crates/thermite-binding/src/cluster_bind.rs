use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::cluster::{KMeans as CoreKMeans, DBSCAN as CoreDBSCAN};
use thermite_core::sparse::build_csr;
#[pyclass]
pub struct KMeans {
    core: CoreKMeans,
}

#[pymethods]
impl KMeans {
    #[new]
    #[pyo3(signature = (n_clusters=8, max_iter=300, tol=1e-4, n_init=10, random_state=None))]
    fn new(
        n_clusters: usize,
        max_iter: usize,
        tol: f64,
        n_init: usize,
        random_state: Option<u64>,
    ) -> Self {
        KMeans {
            core: CoreKMeans::new(n_clusters, max_iter, tol, n_init, random_state),
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

    #[pyo3(signature = (data, indices, indptr, rows, cols))]
    fn fit_sparse(
        &mut self,
        data: PyReadonlyArray1<f64>,
        indices: PyReadonlyArray1<usize>,
        indptr: PyReadonlyArray1<usize>,
        rows: usize,
        cols: usize,
    ) -> PyResult<()> {
        let data_slice = data.as_slice().unwrap();
        let indices_slice = indices.as_slice().unwrap();
        let indptr_slice = indptr.as_slice().unwrap();
        
        let cs_mat = build_csr(data_slice, indices_slice, indptr_slice, rows, cols)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            
        self.core.fit_sparse(&cs_mat).map_err(pyo3::exceptions::PyValueError::new_err)
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
    ) -> PyResult<Bound<'py, PyArray1<usize>>> {
        let data_slice = data.as_slice().unwrap();
        let indices_slice = indices.as_slice().unwrap();
        let indptr_slice = indptr.as_slice().unwrap();
        let cs_mat = build_csr(data_slice, indices_slice, indptr_slice, rows, cols)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            
        let preds = self.core.predict_sparse(&cs_mat).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_vec_bound(py, preds))
    }
    fn fit_predict<'py>(&mut self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<usize>>> {
        let preds = self.core.fit_predict(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_vec_bound(py, preds))
    }

    #[getter]
    fn cluster_centers_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match &self.core.cluster_centers_ {
            Some(c) => Ok(Some(PyArray2::from_array_bound(py, c))),
            None => Ok(None),
        }
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<usize>>>> {
        match &self.core.labels_ {
            Some(l) => Ok(Some(PyArray1::from_vec_bound(py, l.clone()))),
            None => Ok(None),
        }
    }

    #[getter]
    fn inertia_(&self) -> Option<f64> {
        self.core.inertia_
    }

    #[getter]
    fn n_iter_(&self) -> Option<usize> {
        self.core.n_iter_
    }
}

#[pyclass]
pub struct DBSCAN {
    core: CoreDBSCAN,
}

#[pymethods]
impl DBSCAN {
    #[new]
    #[pyo3(signature = (eps=0.5, min_samples=5))]
    fn new(eps: f64, min_samples: usize) -> Self {
        DBSCAN {
            core: CoreDBSCAN::new(eps, min_samples),
        }
    }

    fn fit(&mut self, py: Python<'_>, X: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_view = X.as_array();
        py.allow_threads(|| {
            self.core.fit(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)
        })
    }

    fn fit_predict<'py>(&mut self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let preds = self.core.fit_predict(&X.as_array()).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_vec_bound(py, preds))
    }

    #[getter]
    fn labels_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<i64>>>> {
        match &self.core.labels_ {
            Some(l) => Ok(Some(PyArray1::from_vec_bound(py, l.clone()))),
            None => Ok(None),
        }
    }

    #[getter]
    fn core_sample_indices_<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<usize>>>> {
        match &self.core.core_sample_indices_ {
            Some(i) => Ok(Some(PyArray1::from_vec_bound(py, i.clone()))),
            None => Ok(None),
        }
    }
}

pub fn bind_cluster(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KMeans>()?;
    m.add_class::<DBSCAN>()?;
    Ok(())
}
