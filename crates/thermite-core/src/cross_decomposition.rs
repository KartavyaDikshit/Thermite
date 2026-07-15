use ndarray::{Array1, Array2, ArrayView2, Axis};

fn power_svd(A: &Array2<f64>, n_components: usize, max_iter: usize) -> (Array2<f64>, Array1<f64>, Array2<f64>) {
    let m = A.nrows();
    let n = A.ncols();
    let k = n_components.min(m).min(n);
    let mut U = Array2::<f64>::zeros((m, k));
    let mut S = Array1::<f64>::zeros(k);
    let mut Vt = Array2::<f64>::zeros((k, n));

    let mut work = A.to_owned();
    for comp in 0..k {
        let mut v = Array1::from_shape_fn(n, |_| rand::random::<f64>() - 0.5);
        let v_norm = v.dot(&v).sqrt();
        if v_norm > 0.0 { v /= v_norm; }

        for _ in 0..100 {
            let u = work.dot(&v);
            let u_norm = u.dot(&u).sqrt();
            if u_norm < 1e-12 { break; }
            let u = u / u_norm;
            let v_new = work.t().dot(&u);
            let v_new_norm = v_new.dot(&v_new).sqrt();
            if v_new_norm < 1e-12 { break; }
            let v_new = v_new / v_new_norm;
            let diff = (&v_new - &v).mapv(|x| x.abs()).sum();
            v = v_new;
            if diff < 1e-8 { break; }
        }

        let u = work.dot(&v);
        let sigma = u.dot(&u).sqrt();
        if sigma < 1e-12 { break; }
        let u = u / sigma;

        U.column_mut(comp).assign(&u);
        S[comp] = sigma;
        Vt.row_mut(comp).assign(&v);

        work = &work - sigma * u.dot(&v.t());
    }

    (U, S, Vt)
}

pub struct PLSRegression {
    pub n_components: usize,
    pub scale: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub copy: bool,
    pub x_weights_: Option<Array2<f64>>,
    pub y_weights_: Option<Array2<f64>>,
    pub x_loadings_: Option<Array2<f64>>,
    pub y_loadings_: Option<Array2<f64>>,
    pub x_scores_: Option<Array2<f64>>,
    pub y_scores_: Option<Array2<f64>>,
    pub x_rotations_: Option<Array2<f64>>,
    pub y_rotations_: Option<Array2<f64>>,
    pub coef_: Option<Array2<f64>>,
    pub x_mean_: Option<Array1<f64>>,
    pub y_mean_: Option<Array1<f64>>,
    pub x_std_: Option<Array1<f64>>,
    pub y_std_: Option<Array1<f64>>,
}

impl PLSRegression {
    pub fn new(n_components: usize, scale: bool, max_iter: usize, tol: f64, copy: bool) -> Self {
        Self {
            n_components,
            scale,
            max_iter,
            tol,
            copy,
            x_weights_: None,
            y_weights_: None,
            x_loadings_: None,
            y_loadings_: None,
            x_scores_: None,
            y_scores_: None,
            x_rotations_: None,
            y_rotations_: None,
            coef_: None,
            x_mean_: None,
            y_mean_: None,
            x_std_: None,
            y_std_: None,
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Result<(), String> {
        let n = X.nrows();
        let p = X.ncols();
        let q = y.ncols();

        let mut X_centered = X.to_owned();
        let mut y_centered = y.to_owned();
        let x_mean = X.mean_axis(Axis(0)).unwrap();
        let y_mean = y.mean_axis(Axis(0)).unwrap();
        for mut row in X_centered.rows_mut() { row -= &x_mean; }
        for mut row in y_centered.rows_mut() { row -= &y_mean; }

        let mut x_weights = Array2::<f64>::zeros((p, self.n_components));
        let mut y_weights = Array2::<f64>::zeros((q, self.n_components));
        let mut x_loadings = Array2::<f64>::zeros((p, self.n_components));
        let mut y_loadings = Array2::<f64>::zeros((q, self.n_components));
        let mut x_scores = Array2::<f64>::zeros((n, self.n_components));
        let mut y_scores = Array2::<f64>::zeros((n, self.n_components));

        let mut Xk = X_centered.clone();
        let mut Yk = y_centered.clone();

        for k in 0..self.n_components {
            let mut y_score = Yk.column(0).to_owned();
            let mut w = Array1::<f64>::zeros(p);
            let mut c = Array1::<f64>::zeros(q);

            for _ in 0..self.max_iter {
                let u_norm = y_score.dot(&y_score);
                if u_norm < 1e-12 { break; }
                w = Xk.t().dot(&y_score) / u_norm;
                let w_norm = w.dot(&w).sqrt();
                if w_norm < 1e-12 { break; }
                w /= w_norm;

                let t = Xk.dot(&w);
                let t_norm = t.dot(&t);
                if t_norm < 1e-12 { break; }

                c = Yk.t().dot(&t) / t_norm;
                let c_norm = c.dot(&c);
                if c_norm < 1e-12 { break; }
                let u = Yk.dot(&c) / c_norm;

                let diff = (&u - &y_score).mapv(|v| v.abs()).sum();
                y_score = u;
                if diff < self.tol { break; }
            }

            let t = Xk.dot(&w);
            let t_norm = t.dot(&t);
            if t_norm < 1e-12 { break; }
            let p = Xk.t().dot(&t) / t_norm;

            let u_norm = y_score.dot(&y_score);
            if u_norm < 1e-12 { break; }
            let q = Yk.t().dot(&y_score) / u_norm;

            x_weights.column_mut(k).assign(&w);
            y_weights.column_mut(k).assign(&c);
            x_loadings.column_mut(k).assign(&p);
            y_loadings.column_mut(k).assign(&q);
            x_scores.column_mut(k).assign(&t);

            let t = Xk.dot(&w);
            let p = Xk.t().dot(&t) / t.dot(&t);
            Xk = &Xk - t.dot(&p.t());
            let u = Yk.dot(&c);
            let q = Yk.t().dot(&u) / u.dot(&u);
            Yk = &Yk - u.dot(&q.t());
        }

        let coef = X_centered.t().dot(&X_centered);
        let coef = power_svd(&coef, coef.ncols().min(coef.nrows()), 100);
        let coef = coef.0.dot(&Array2::from_diag(&coef.1.mapv(|v| if v > 1e-12 { 1.0 / v } else { 0.0 }))).dot(&coef.2);
        let coef = coef.dot(&X_centered.t()).dot(&y_centered);

        self.x_weights_ = Some(x_weights);
        self.y_weights_ = Some(y_weights);
        self.x_loadings_ = Some(x_loadings);
        self.y_loadings_ = Some(y_loadings);
        self.x_scores_ = Some(x_scores);
        self.y_scores_ = Some(y_scores);
        self.coef_ = Some(coef);
        self.x_mean_ = Some(x_mean);
        self.y_mean_ = Some(y_mean);

        Ok(())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        if let Some(coef) = &self.coef_ {
            Ok(X.dot(coef))
        } else {
            Err("Model not fitted".to_string())
        }
    }
}

pub struct CCA {
    pub n_components: usize,
    pub scale: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub copy: bool,
    pub x_weights_: Option<Array2<f64>>,
    pub y_weights_: Option<Array2<f64>>,
    pub x_loadings_: Option<Array2<f64>>,
    pub y_loadings_: Option<Array2<f64>>,
    pub x_scores_: Option<Array2<f64>>,
    pub y_scores_: Option<Array2<f64>>,
    pub x_rotations_: Option<Array2<f64>>,
    pub y_rotations_: Option<Array2<f64>>,
    pub coef_: Option<Array2<f64>>,
}

impl CCA {
    pub fn new(n_components: usize, scale: bool, max_iter: usize, tol: f64, copy: bool) -> Self {
        Self {
            n_components,
            scale,
            max_iter,
            tol,
            copy,
            x_weights_: None,
            y_weights_: None,
            x_loadings_: None,
            y_loadings_: None,
            x_scores_: None,
            y_scores_: None,
            x_rotations_: None,
            y_rotations_: None,
            coef_: None,
        }
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView2<f64>) -> Result<(), String> {
        let p = X.ncols();
        let q = y.ncols();

        let x_mean = X.mean_axis(Axis(0)).unwrap();
        let y_mean = y.mean_axis(Axis(0)).unwrap();
        let mut Xc = X.to_owned();
        let mut Yc = y.to_owned();
        for mut row in Xc.rows_mut() { row -= &x_mean; }
        for mut row in Yc.rows_mut() { row -= &y_mean; }

        let C = Xc.t().dot(&Yc);
        let k = self.n_components.min(p).min(q);
        let (U, _S, Vt) = power_svd(&C, k, self.max_iter);

        let x_weights = U;
        let y_weights = Vt.t().to_owned();
        let x_scores = Xc.dot(&x_weights);
        let y_scores = Yc.dot(&y_weights);
        let coef = x_weights.dot(&y_weights.t());

        self.x_weights_ = Some(x_weights);
        self.y_weights_ = Some(y_weights);
        self.x_scores_ = Some(x_scores);
        self.y_scores_ = Some(y_scores);
        self.coef_ = Some(coef);

        Ok(())
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        if let Some(coef) = &self.coef_ {
            Ok(X.dot(coef))
        } else {
            Err("Model not fitted".to_string())
        }
    }
}
