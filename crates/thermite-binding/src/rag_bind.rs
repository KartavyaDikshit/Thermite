use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use thermite_core::rag::VectorStore as CoreVectorStore;

#[pyclass]
pub struct VectorStore {
    core: CoreVectorStore,
}

#[pymethods]
impl VectorStore {
    #[new]
    fn new(vectors: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let arr = vectors.as_array().to_owned();
        Ok(VectorStore {
            core: CoreVectorStore::new(arr),
        })
    }

    fn search<'py>(&self, py: Python<'py>, query: PyReadonlyArray1<f64>, k: usize) -> PyResult<Bound<'py, PyTuple>> {
        let query_arr = query.as_array().to_owned();
        let (indices, distances) = self.core.search(&query_arr, k).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let py_indices = PyArray1::from_vec_bound(py, indices.into_iter().map(|x| x as i64).collect());
        let py_distances = PyArray1::from_vec_bound(py, distances);
        Ok(PyTuple::new_bound(py, [py_indices.as_any(), py_distances.as_any()]))
    }
}

pub fn bind_rag(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VectorStore>()?;
    Ok(())
}
