use ndarray::{Array2, ShapeBuilder};
fn main() {
    let a = Array2::<f64>::zeros((10, 20).f());
    println!("{:?}", a.strides());
}
