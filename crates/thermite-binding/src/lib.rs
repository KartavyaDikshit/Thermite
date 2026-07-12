#![allow(non_snake_case)]
use pyo3::prelude::*;
use pyo3::types::{PyTuple, PyList};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};


use thermite_core::model_selection::compute_split_indices;
use thermite_core::preprocessing::StandardScaler as CoreStandardScaler;
use thermite_core::preprocessing::MinMaxScaler as CoreMinMaxScaler;
use thermite_core::preprocessing::LabelEncoderCore;
use thermite_core::preprocessing::OneHotEncoderCore;

#[pyfunction]
fn ping() -> PyResult<String> {
    Ok("pong".to_string())
}

#[pyfunction]
#[pyo3(signature = (*arrays, test_size=None, train_size=None, random_state=None, shuffle=true, stratify=None))]
fn train_test_split<'py>(
    py: Python<'py>,
    arrays: &Bound<'py, PyTuple>,
    test_size: Option<f64>,
    train_size: Option<f64>,
    random_state: Option<u64>,
    shuffle: bool,
    stratify: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyTuple>> {
    if arrays.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("At least one array is required as input"));
    }

    let first_arr = arrays.get_item(0)?;
    let len_any: usize = first_arr.len()?;
    
    for i in 1..arrays.len() {
        let arr = arrays.get_item(i)?;
        if arr.len()? != len_any {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All input arrays must have the same length"
            ));
        }
    }

    let stratify_vec = if let Some(strat_obj) = stratify {
        let mut labels = Vec::new();
        let len = strat_obj.len()?;
        if len != len_any {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Stratify labels length must match number of samples"
            ));
        }
        for item in strat_obj.iter()? {
            let val = item?;
            labels.push(val.str()?.to_string());
        }
        Some(labels)
    } else {
        None
    };

    let split = compute_split_indices(
        len_any,
        test_size,
        train_size,
        shuffle,
        random_state,
        stratify_vec.as_deref(),
    ).map_err(pyo3::exceptions::PyValueError::new_err)?;

    let train_indices_i64: Vec<i64> = split.train_indices.iter().map(|&x| x as i64).collect();
    let test_indices_i64: Vec<i64> = split.test_indices.iter().map(|&x| x as i64).collect();

    let train_idx_py = PyArray1::from_vec_bound(py, train_indices_i64);
    let test_idx_py = PyArray1::from_vec_bound(py, test_indices_i64);

    let mut results = Vec::with_capacity(arrays.len() * 2);
    for i in 0..arrays.len() {
        let arr = arrays.get_item(i)?;
        let train_slice = arr.call_method1("__getitem__", (train_idx_py.clone(),))?;
        let test_slice = arr.call_method1("__getitem__", (test_idx_py.clone(),))?;
        results.push(train_slice);
        results.push(test_slice);
    }

    Ok(PyTuple::new_bound(py, results))
}

#[pyclass]
pub struct StandardScaler {
    core: CoreStandardScaler,
}

#[pymethods]
impl StandardScaler {
    #[new]
    #[pyo3(signature = (with_mean=true, with_std=true))]
    fn new(with_mean: bool, with_std: bool) -> Self {
        StandardScaler {
            core: CoreStandardScaler::new(with_mean, with_std),
        }
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_arr = X.as_array();
        self.core.fit(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        let out = self.core.transform(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &out))
    }

    fn fit_transform<'py>(&mut self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        self.core.fit(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let out = self.core.transform(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &out))
    }

    fn inverse_transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        let out = self.core.inverse_transform(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &out))
    }

    #[getter]
    fn mean<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.mean {
            Some(m) => Ok(Some(PyArray1::from_array_bound(py, m))),
            None => Ok(None),
        }
    }

    #[getter]
    fn var<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        if !self.core.with_std {
            return Ok(None);
        }
        match &self.core.var {
            Some(v) => Ok(Some(PyArray1::from_array_bound(py, v))),
            None => Ok(None),
        }
    }

    #[getter]
    fn scale<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        if !self.core.with_std {
            return Ok(None);
        }
        match &self.core.scale {
            Some(s) => Ok(Some(PyArray1::from_array_bound(py, s))),
            None => Ok(None),
        }
    }

    #[getter]
    fn n_samples_seen(&self) -> usize {
        self.core.n_samples_seen
    }
}

#[pyclass]
pub struct MinMaxScaler {
    core: CoreMinMaxScaler,
}

#[pymethods]
impl MinMaxScaler {
    #[new]
    #[pyo3(signature = (feature_range=(0.0, 1.0)))]
    fn new(feature_range: (f64, f64)) -> PyResult<Self> {
        let core = CoreMinMaxScaler::new(feature_range).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(MinMaxScaler { core })
    }

    fn fit(&mut self, X: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x_arr = X.as_array();
        self.core.fit(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        let out = self.core.transform(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &out))
    }

    fn fit_transform<'py>(&mut self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        self.core.fit(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        let out = self.core.transform(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &out))
    }

    fn inverse_transform<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        let out = self.core.inverse_transform(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &out))
    }

    #[getter]
    fn data_min<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.data_min {
            Some(dm) => Ok(Some(PyArray1::from_array_bound(py, dm))),
            None => Ok(None),
        }
    }

    #[getter]
    fn data_max<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.data_max {
            Some(dm) => Ok(Some(PyArray1::from_array_bound(py, dm))),
            None => Ok(None),
        }
    }

    #[getter]
    fn scale<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.scale {
            Some(s) => Ok(Some(PyArray1::from_array_bound(py, s))),
            None => Ok(None),
        }
    }

    #[getter]
    fn min<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
        match &self.core.min {
            Some(m) => Ok(Some(PyArray1::from_array_bound(py, m))),
            None => Ok(None),
        }
    }
}

#[pyclass]
pub struct LabelEncoder {
    core: LabelEncoderCore,
}

#[pymethods]
impl LabelEncoder {
    #[new]
    fn new() -> Self {
        LabelEncoder {
            core: LabelEncoderCore::new(),
        }
    }

    fn fit_int(&mut self, y: PyReadonlyArray1<i64>) {
        let y_view = y.as_array();
        if let Some(slice) = y_view.as_slice() {
            self.core.fit_int(slice);
        } else {
            let vec = y_view.to_vec();
            self.core.fit_int(&vec);
        }
    }

    fn fit_str(&mut self, y: Vec<String>) {
        self.core.fit_str(y);
    }

    fn transform_int<'py>(&self, py: Python<'py>, y: PyReadonlyArray1<i64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let y_view = y.as_array();
        let out = if let Some(slice) = y_view.as_slice() {
            self.core.transform_int(slice)
        } else {
            let vec = y_view.to_vec();
            self.core.transform_int(&vec)
        }.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_vec_bound(py, out))
    }

    fn transform_str<'py>(&self, py: Python<'py>, y: Vec<String>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let out = self.core.transform_str(&y).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_vec_bound(py, out))
    }

    fn inverse_transform_int<'py>(&self, py: Python<'py>, y: PyReadonlyArray1<i64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let y_view = y.as_array();
        let out = if let Some(slice) = y_view.as_slice() {
            self.core.inverse_transform_int(slice)
        } else {
            let vec = y_view.to_vec();
            self.core.inverse_transform_int(&vec)
        }.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_vec_bound(py, out))
    }

    fn inverse_transform_str(&self, y: PyReadonlyArray1<i64>) -> PyResult<Vec<String>> {
        let y_view = y.as_array();
        if let Some(slice) = y_view.as_slice() {
            self.core.inverse_transform_str(slice)
        } else {
            let vec = y_view.to_vec();
            self.core.inverse_transform_str(&vec)
        }.map_err(pyo3::exceptions::PyValueError::new_err)
    }

    fn get_classes_int(&self) -> Option<Vec<i64>> {
        self.core.classes_int.clone()
    }

    fn get_classes_str(&self) -> Option<Vec<String>> {
        self.core.classes_str.clone()
    }

    fn fit_float(&mut self, y: PyReadonlyArray1<f64>) {
        let y_view = y.as_array();
        if let Some(slice) = y_view.as_slice() {
            self.core.fit_float(slice);
        } else {
            let vec = y_view.to_vec();
            self.core.fit_float(&vec);
        }
    }

    fn transform_float<'py>(&self, py: Python<'py>, y: PyReadonlyArray1<f64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let y_view = y.as_array();
        let out = if let Some(slice) = y_view.as_slice() {
            self.core.transform_float(slice)
        } else {
            let vec = y_view.to_vec();
            self.core.transform_float(&vec)
        }.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_vec_bound(py, out))
    }

    fn inverse_transform_float<'py>(&self, py: Python<'py>, y: PyReadonlyArray1<i64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let y_view = y.as_array();
        let out = if let Some(slice) = y_view.as_slice() {
            self.core.inverse_transform_float(slice)
        } else {
            let vec = y_view.to_vec();
            self.core.inverse_transform_float(&vec)
        }.map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray1::from_vec_bound(py, out))
    }

    fn get_classes_float(&self) -> Option<Vec<f64>> {
        self.core.classes_float.clone()
    }
}

#[pyclass]
pub struct OneHotEncoder {
    core: OneHotEncoderCore,
}

#[pymethods]
impl OneHotEncoder {
    #[new]
    #[pyo3(signature = (handle_unknown="error"))]
    fn new(handle_unknown: &str) -> PyResult<Self> {
        if handle_unknown != "error" && handle_unknown != "ignore" {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "handle_unknown must be either 'error' or 'ignore'"
            ));
        }
        Ok(OneHotEncoder {
            core: OneHotEncoderCore::new(handle_unknown.to_string()),
        })
    }

    fn fit_int(&mut self, X: PyReadonlyArray2<i64>) {
        let x_arr = X.as_array();
        self.core.fit_int(&x_arr);
    }

    fn fit_str(&mut self, X_cols: Vec<Vec<String>>) {
        self.core.fit_str(X_cols);
    }

    fn transform_int<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<i64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = X.as_array();
        let out = self.core.transform_int(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &out))
    }

    fn transform_str<'py>(&self, py: Python<'py>, X_cols: Vec<Vec<String>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let out = self.core.transform_str(&X_cols).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &out))
    }

    fn inverse_transform_int<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let x_arr = X.as_array();
        let out = self.core.inverse_transform_int(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(PyArray2::from_array_bound(py, &out))
    }

    fn inverse_transform_str(&self, X: PyReadonlyArray2<f64>) -> PyResult<Vec<Vec<String>>> {
        let x_arr = X.as_array();
        self.core.inverse_transform_str(&x_arr).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    #[getter]
    fn categories<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty_bound(py);
        for cat in &self.core.categories {
            match cat {
                thermite_core::preprocessing::ColumnCategories::Int(cats) => {
                    list.append(PyArray1::from_vec_bound(py, cats.clone()))?;
                }
                thermite_core::preprocessing::ColumnCategories::Str(cats) => {
                    list.append(cats.clone())?;
                }
            }
        }
        Ok(list)
    }
}

pub mod linear_model_bind;
pub mod metrics_bind;
pub mod tree_bind;
pub mod cluster_bind;
pub mod decomposition_bind;
pub mod neighbors_bind;
pub mod ensemble_bind;
pub mod naive_bayes_bind;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(train_test_split, m)?)?;
    m.add_class::<StandardScaler>()?;
    m.add_class::<MinMaxScaler>()?;
    m.add_class::<LabelEncoder>()?;
    m.add_class::<OneHotEncoder>()?;
    linear_model_bind::bind_linear_model(m)?;
    metrics_bind::bind_metrics(m)?;
    tree_bind::bind_tree(m)?;
    cluster_bind::bind_cluster(m)?;
    decomposition_bind::bind_decomposition(m)?;
    neighbors_bind::bind_neighbors(m)?;
    ensemble_bind::bind_ensemble(m)?;
    naive_bayes_bind::bind_naive_bayes(m)?;
    Ok(())
}
