use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use thermite_core::impute::IterativeImputer as CoreIterativeImputer;

#[pyclass]
pub struct IterativeImputer {
    core: CoreIterativeImputer,
}

#[pymethods]
impl IterativeImputer {
    #[new]
    #[pyo3(signature = (max_iter=10))]
    fn new(max_iter: usize) -> Self {
        IterativeImputer {
            core: CoreIterativeImputer::new(max_iter),
        }
    }

    fn fit_transform<'py>(&mut self, py: Python<'py>, X: PyReadonlyArray2<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_view = X.as_array();
        let out = self.core.fit_transform(&x_view).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(out.to_pyarray_bound(py))
    }
}

pub fn bind_impute(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IterativeImputer>()?;
    Ok(())
}
