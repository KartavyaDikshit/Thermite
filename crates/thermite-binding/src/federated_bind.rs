use pyo3::prelude::*;
use thermite_core::federated::ParameterServer as CoreParameterServer;
use crate::linear_model_bind::SGDClassifier;

#[pyclass]
pub struct ParameterServer {
    core: CoreParameterServer,
}

#[pymethods]
impl ParameterServer {
    #[new]
    fn new() -> Self {
        ParameterServer {
            core: CoreParameterServer::new(),
        }
    }

    fn aggregate(&self, models: Vec<PyRef<SGDClassifier>>) -> PyResult<SGDClassifier> {
        let core_models: Vec<&thermite_core::linear_model::SGDClassifier> = models.iter().map(|m| &m.core).collect();
        let agg_core = self.core.aggregate(&core_models).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(SGDClassifier { core: agg_core })
    }
}

pub fn bind_federated(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ParameterServer>()?;
    Ok(())
}
