use ndarray::{Array1, Array2};
fn main() {
    let X = Array2::<f64>::zeros((1000, 10));
    let Xt = X.t().to_owned();
    println!("Xt shape: {:?}", Xt.shape());
}
