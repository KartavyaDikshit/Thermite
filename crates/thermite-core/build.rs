fn main() {
    cc::Build::new()
        .cpp(true)
        .file("src/libsvm/svm.cpp")
        .compile("libsvm");
    println!("cargo:rerun-if-changed=src/libsvm/svm.cpp");
    println!("cargo:rerun-if-changed=src/libsvm/svm.h");
}
