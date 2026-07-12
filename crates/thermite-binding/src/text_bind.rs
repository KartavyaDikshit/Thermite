use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray2, ToPyArray};
use thermite_core::text::{CountVectorizer as CoreCountVectorizer, TfidfVectorizer as CoreTfidfVectorizer};

#[pyclass]
pub struct CountVectorizer {
    core: CoreCountVectorizer,
}

#[pymethods]
impl CountVectorizer {
    #[new]
    #[pyo3(signature = (lowercase=true))]
    fn new(lowercase: bool) -> Self {
        CountVectorizer {
            core: CoreCountVectorizer::new(lowercase),
        }
    }

    fn fit(&mut self, docs: Vec<String>) -> PyResult<()> {
        self.core.fit(&docs);
        Ok(())
    }

    fn transform<'py>(&self, py: Python<'py>, docs: Vec<String>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let out = self.core.transform(&docs);
        Ok(out.to_pyarray_bound(py))
    }

    fn fit_transform<'py>(&mut self, py: Python<'py>, docs: Vec<String>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let out = self.core.fit_transform(&docs);
        Ok(out.to_pyarray_bound(py))
    }

    #[getter]
    fn vocabulary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        for (k, v) in &self.core.vocabulary {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }
}

#[pyclass]
pub struct TfidfVectorizer {
    core: CoreTfidfVectorizer,
}

#[pymethods]
impl TfidfVectorizer {
    #[new]
    #[pyo3(signature = (lowercase=true))]
    fn new(lowercase: bool) -> Self {
        TfidfVectorizer {
            core: CoreTfidfVectorizer::new(lowercase),
        }
    }

    fn fit(&mut self, docs: Vec<String>) -> PyResult<()> {
        self.core.fit(&docs);
        Ok(())
    }

    fn transform<'py>(&self, py: Python<'py>, docs: Vec<String>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let out = self.core.transform(&docs);
        Ok(out.to_pyarray_bound(py))
    }

    fn fit_transform<'py>(&mut self, py: Python<'py>, docs: Vec<String>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let out = self.core.fit_transform(&docs);
        Ok(out.to_pyarray_bound(py))
    }

    #[getter]
    fn vocabulary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        for (k, v) in &self.core.count_vec.vocabulary {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }

    #[getter]
    fn idf(&self) -> Vec<f64> {
        self.core.idf.clone()
    }
}

pub fn bind_text(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CountVectorizer>()?;
    m.add_class::<TfidfVectorizer>()?;
    m.add_class::<Word2Vec>()?;
    Ok(())
}

use thermite_core::text::Word2Vec as CoreWord2Vec;

#[pyclass]
pub struct Word2Vec {
    core: CoreWord2Vec,
}

#[pymethods]
impl Word2Vec {
    #[new]
    #[pyo3(signature = (vector_size=100, window=5, min_count=5))]
    fn new(vector_size: usize, window: usize, min_count: usize) -> Self {
        Word2Vec {
            core: CoreWord2Vec::new(vector_size, window, min_count),
        }
    }

    fn fit(&mut self, sentences: Vec<Vec<String>>) -> PyResult<()> {
        self.core.fit(&sentences).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(())
    }

    #[getter]
    fn embeddings<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        match self.core.get_embeddings() {
            Ok(emb) => {
                let dict = PyDict::new_bound(py);
                for (k, v) in emb {
                    let arr = numpy::PyArray1::from_vec_bound(py, v);
                    dict.set_item(k, arr)?;
                }
                Ok(Some(dict))
            }
            Err(_) => Ok(None),
        }
    }
}
