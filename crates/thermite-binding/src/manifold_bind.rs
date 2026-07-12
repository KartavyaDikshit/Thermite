use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use thermite_core::manifold::{TSNE, UMAP};

#[pyclass(name = "TSNE")]
pub struct PyTSNE {
    inner: TSNE,
}

#[pymethods]
impl PyTSNE {
    #[new]
    #[pyo3(signature = (n_components=2, perplexity=30.0))]
    fn new(n_components: usize, perplexity: f64) -> Self {
        PyTSNE {
            inner: TSNE::new(n_components, perplexity),
        }
    }

    fn fit_transform<'py>(&mut self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let x_shape = x.shape();
        let x_slice = x.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("X is not contiguous"))?;
        let res = self.inner.fit_transform(x_slice, x_shape[0], x_shape[1])
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        
        use numpy::ndarray::Array2;
        let res_array = Array2::from_shape_vec((x_shape[0], self.inner.n_components), res)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(res_array.into_pyarray_bound(py).into_any())
    }
}

#[pyclass(name = "UMAP")]
pub struct PyUMAP {
    inner: UMAP,
}

#[pymethods]
impl PyUMAP {
    #[new]
    #[pyo3(signature = (n_components=2, n_neighbors=15))]
    fn new(n_components: usize, n_neighbors: usize) -> Self {
        PyUMAP {
            inner: UMAP::new(n_components, n_neighbors),
        }
    }

    fn fit_transform<'py>(&mut self, py: Python<'py>, x: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let x_shape = x.shape();
        let x_slice = x.as_slice().map_err(|_| pyo3::exceptions::PyValueError::new_err("X is not contiguous"))?;
        let res = self.inner.fit_transform(x_slice, x_shape[0], x_shape[1])
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        
        use numpy::ndarray::Array2;
        let res_array = Array2::from_shape_vec((x_shape[0], self.inner.n_components), res)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(res_array.into_pyarray_bound(py).into_any())
    }
}

pub fn bind_manifold(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTSNE>()?;
    m.add_class::<PyUMAP>()?;
    Ok(())
}
