fn main() {
    cc::Build::new()
        .cpp(true)
        .flag_if_supported("-std=c++14")
        .flag_if_supported("-std=c++11")
        .file("src/libsvm/svm.cpp")
        .compile("libsvm");
    println!("cargo:rerun-if-changed=src/libsvm/svm.cpp");
    println!("cargo:rerun-if-changed=src/libsvm/svm.h");
}
