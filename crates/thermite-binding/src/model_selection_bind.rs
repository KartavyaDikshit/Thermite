use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyReadonlyArray1};
use thermite_core::model_selection::{StratifiedKFold as CoreStratifiedKFold, TimeSeriesSplit as CoreTimeSeriesSplit, GroupKFold as CoreGroupKFold};

#[pyclass]
pub struct StratifiedKFold {
    core: CoreStratifiedKFold,
}

#[pymethods]
impl StratifiedKFold {
    #[new]
    #[pyo3(signature = (n_splits=5, shuffle=false, random_state=None))]
    fn new(n_splits: usize, shuffle: bool, random_state: Option<u64>) -> Self {
        StratifiedKFold {
            core: CoreStratifiedKFold {
                n_splits,
                shuffle,
                random_state,
            }
        }
    }

    fn split<'py>(&self, py: Python<'py>, X: Bound<'py, PyAny>, y: PyReadonlyArray1<i64>) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        let y_arr = y.as_array();
        let y_slice = y_arr.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("y must be contiguous"))?;
        let splits = self.core.split(y_slice).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let mut py_splits = Vec::new();
        for split in splits {
            let train_indices: Vec<i64> = split.train_indices.iter().map(|&x| x as i64).collect();
            let test_indices: Vec<i64> = split.test_indices.iter().map(|&x| x as i64).collect();
            let py_train = PyArray1::from_vec_bound(py, train_indices);
            let py_test = PyArray1::from_vec_bound(py, test_indices);
            py_splits.push(PyTuple::new_bound(py, [py_train.as_any(), py_test.as_any()]));
        }
        Ok(py_splits)
    }
}

#[pyclass]
pub struct TimeSeriesSplit {
    core: CoreTimeSeriesSplit,
}

#[pymethods]
impl TimeSeriesSplit {
    #[new]
    #[pyo3(signature = (n_splits=5))]
    fn new(n_splits: usize) -> Self {
        TimeSeriesSplit {
            core: CoreTimeSeriesSplit {
                n_splits,
            }
        }
    }

    fn split<'py>(&self, py: Python<'py>, X: Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        let n_samples = X.len()?;
        let splits = self.core.split(n_samples).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let mut py_splits = Vec::new();
        for split in splits {
            let train_indices: Vec<i64> = split.train_indices.iter().map(|&x| x as i64).collect();
            let test_indices: Vec<i64> = split.test_indices.iter().map(|&x| x as i64).collect();
            let py_train = PyArray1::from_vec_bound(py, train_indices);
            let py_test = PyArray1::from_vec_bound(py, test_indices);
            py_splits.push(PyTuple::new_bound(py, [py_train.as_any(), py_test.as_any()]));
        }
        Ok(py_splits)
    }
}

#[pyclass]
pub struct GroupKFold {
    core: CoreGroupKFold,
}

#[pymethods]
impl GroupKFold {
    #[new]
    #[pyo3(signature = (n_splits=5))]
    fn new(n_splits: usize) -> Self {
        GroupKFold {
            core: CoreGroupKFold {
                n_splits,
            }
        }
    }

    fn split<'py>(&self, py: Python<'py>, X: Bound<'py, PyAny>, y: Bound<'py, PyAny>, groups: PyReadonlyArray1<i64>) -> PyResult<Vec<Bound<'py, PyTuple>>> {
        let g_arr = groups.as_array();
        let g_slice = g_arr.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("groups must be contiguous"))?;
        let splits = self.core.split(g_slice).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let mut py_splits = Vec::new();
        for split in splits {
            let train_indices: Vec<i64> = split.train_indices.iter().map(|&x| x as i64).collect();
            let test_indices: Vec<i64> = split.test_indices.iter().map(|&x| x as i64).collect();
            let py_train = PyArray1::from_vec_bound(py, train_indices);
            let py_test = PyArray1::from_vec_bound(py, test_indices);
            py_splits.push(PyTuple::new_bound(py, [py_train.as_any(), py_test.as_any()]));
        }
        Ok(py_splits)
    }
}

pub fn bind_model_selection(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StratifiedKFold>()?;
    m.add_class::<TimeSeriesSplit>()?;
    m.add_class::<GroupKFold>()?;
    Ok(())
}
