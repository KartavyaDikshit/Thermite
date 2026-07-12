#ifndef THERMITE_SVM_H
#define THERMITE_SVM_H

#ifdef __cplusplus
extern "C" {
#endif

struct svm_model {
    int kernel_type;
    int degree;
    double gamma;
    double coef0;
    
    int l;               // number of support vectors
    int D;               // dimension of features
    double* SVs;         // support vectors (flat array of size l * D)
    double* sv_coef;     // coefficients alpha_i * y_i (size l)
    double rho;          // negative intercept (-b)
    
    double probA;        // Platt parameter A
    double probB;        // Platt parameter B
};

svm_model* svm_train(
    const double* X, 
    const double* y, 
    int N, 
    int D,
    double C, 
    int kernel_type, 
    int degree, 
    double gamma, 
    double coef0,
    double eps, 
    int max_iter,
    int probability
);

double svm_predict(const svm_model* model, const double* x);
double svm_predict_decision(const svm_model* model, const double* x);
void svm_free_model(svm_model* model);

#ifdef __cplusplus
}
#endif

#endif // THERMITE_SVM_H
