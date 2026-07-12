use crate::linear_model::SGDClassifier;

pub struct ParameterServer {}

impl ParameterServer {
    pub fn new() -> Self {
        ParameterServer {}
    }

    pub fn aggregate(&self, models: &[&SGDClassifier]) -> Result<SGDClassifier, String> {
        if models.is_empty() {
            return Err("No models to aggregate".to_string());
        }

        let first = models[0];
        if first.coef_.is_none() || first.intercept_.is_none() {
            return Err("Models are not fitted".to_string());
        }

        let n = models.len() as f64;
        let mut sum_coef = first.coef_.as_ref().unwrap().clone();
        let mut sum_intercept = first.intercept_.as_ref().unwrap().clone();

        for i in 1..models.len() {
            let model = models[i];
            if model.coef_.is_none() || model.intercept_.is_none() {
                return Err("Some models are not fitted".to_string());
            }
            sum_coef = sum_coef + model.coef_.as_ref().unwrap();
            sum_intercept = sum_intercept + model.intercept_.as_ref().unwrap();
        }

        let avg_coef = sum_coef / n;
        let avg_intercept = sum_intercept / n;

        let agg_model = SGDClassifier {
            loss: first.loss.clone(),
            penalty: first.penalty.clone(),
            alpha: first.alpha,
            l1_ratio: first.l1_ratio,
            fit_intercept: first.fit_intercept,
            max_iter: first.max_iter,
            tol: first.tol,
            learning_rate: first.learning_rate,
            coef_: Some(avg_coef),
            intercept_: Some(avg_intercept),
            classes_: first.classes_.clone(),
        };

        Ok(agg_model)
    }
}
