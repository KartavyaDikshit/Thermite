use sprs::CsMat;
use ndarray::Array1;

pub fn build_csr(
    data: &[f64],
    indices: &[usize],
    indptr: &[usize],
    rows: usize,
    cols: usize,
) -> Result<CsMat<f64>, String> {
    let indptr_owned = indptr.to_vec();
    let indices_owned = indices.to_vec();
    let data_owned = data.to_vec();
    
    // Check lengths
    if indptr.len() != rows + 1 {
        return Err(format!("indptr length {} != rows + 1 ({})", indptr.len(), rows + 1));
    }
    
    // sprs::CsMat::new is for CSR when the layout is passed, but new is deprecated sometimes or CsMatI::new is used
    Ok(CsMat::new((rows, cols), indptr_owned, indices_owned, data_owned))
}
