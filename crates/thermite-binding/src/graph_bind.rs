use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{IntoPyArray, PyArray2};
use std::collections::HashMap;
use thermite_core::graph::Node2Vec as CoreNode2Vec;

#[pyclass]
pub struct Node2Vec {
    core: CoreNode2Vec,
}

#[pymethods]
impl Node2Vec {
    #[new]
    #[pyo3(signature = (p=1.0, q=1.0, walk_length=80, num_walks=10, embedding_dim=128))]
    fn new(p: f64, q: f64, walk_length: usize, num_walks: usize, embedding_dim: usize) -> Self {
        Node2Vec {
            core: CoreNode2Vec::new(p, q, walk_length, num_walks, embedding_dim),
        }
    }

    fn fit(&mut self, adjacency_list_dict: &Bound<'_, PyDict>) -> PyResult<()> {
        let mut adjacency_list = HashMap::new();
        for (k, v) in adjacency_list_dict {
            let key: usize = k.extract()?;
            let val: Vec<usize> = v.extract()?;
            adjacency_list.insert(key, val);
        }
        self.core.fit(&adjacency_list).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(())
    }

    #[getter]
    fn embeddings<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        match self.core.get_embeddings() {
            Ok(emb) => Ok(Some(emb.into_pyarray_bound(py))),
            Err(_) => Ok(None),
        }
    }
}

pub fn bind_graph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Node2Vec>()?;
    Ok(())
}
