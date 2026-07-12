#include "svm.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>

// Helper to calculate kernel function
double kernel_function(const double* x1, const double* x2, int D, int kernel_type, double gamma, double coef0, int degree) {
    if (kernel_type == 0) { // RBF: exp(-gamma * ||x1 - x2||^2)
        double sum = 0.0;
        for (int i = 0; i < D; ++i) {
            double diff = x1[i] - x2[i];
            sum += diff * diff;
        }
        return std::exp(-gamma * sum);
    } else if (kernel_type == 1) { // Polynomial: (gamma * <x1, x2> + coef0)^degree
        double dot = 0.0;
        for (int i = 0; i < D; ++i) {
            dot += x1[i] * x2[i];
        }
        return std::pow(gamma * dot + coef0, degree);
    }
    return 0.0;
}

// Fit Platt scaling parameters A and B using Newton's method
void fit_platt_scaling(const std::vector<double>& dec_values, const double* y, int N, double& probA, double& probB) {
    int N_plus = 0;
    int N_minus = 0;
    for (int i = 0; i < N; ++i) {
        if (y[i] > 0) N_plus++;
        else N_minus++;
    }
    
    // Set target probabilities (Platt's laplace correction)
    double t_plus = (N_plus + 1.0) / (N_plus + 2.0);
    double t_minus = 1.0 / (N_minus + 2.0);
    
    std::vector<double> t(N);
    for (int i = 0; i < N; ++i) {
        t[i] = (y[i] > 0) ? t_plus : t_minus;
    }
    
    // Initialize A = 0, B = log((N_minus + 1) / (N_plus + 1))
    double A = 0.0;
    double B = std::log((N_minus + 1.0) / (N_plus + 1.0));
    
    int max_newton_iter = 100;
    double eps = 1e-7;
    
    for (int iter = 0; iter < max_newton_iter; ++iter) {
        double gA = 0.0;
        double gB = 0.0;
        double HAA = 0.0;
        double HAB = 0.0;
        double HBB = 0.0;
        
        for (int i = 0; i < N; ++i) {
            double f = dec_values[i];
            double arg = A * f + B;
            double exp_val = std::exp(std::max(-50.0, std::min(50.0, arg)));
            double p = 1.0 / (1.0 + exp_val);
            
            double d = t[i] - p;
            gA += d * f;
            gB += d;
            
            double p_1_p = p * (1.0 - p);
            HAA += p_1_p * f * f;
            HAB += p_1_p * f;
            HBB += p_1_p;
        }
        
        // Add a small diagonal regularizer to prevent singularity
        HAA += 1e-5;
        HBB += 1e-5;
        
        double det = HAA * HBB - HAB * HAB;
        if (std::abs(det) < 1e-12) {
            break;
        }
        
        double dA = (HBB * gA - HAB * gB) / det;
        double dB = (-HAB * gA + HAA * gB) / det;
        
        A -= dA;
        B -= dB;
        
        if (std::abs(dA) < eps && std::abs(dB) < eps) {
            break;
        }
    }
    
    probA = A;
    probB = B;
}

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
) {
    // 1. Precompute kernel matrix to make training fast
    std::vector<double> K(N * N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            K[i * N + j] = kernel_function(&X[i * D], &X[j * D], D, kernel_type, gamma, coef0, degree);
        }
    }
    
    // 2. SMO parameters
    std::vector<double> alpha(N, 0.0);
    double b = 0.0;
    std::vector<double> E(N);
    for (int i = 0; i < N; ++i) {
        E[i] = -y[i];
    }
    
    // 3. SMO core functions
    auto takeStep = [&](int i, int j) -> bool {
        if (i == j) return false;
        double alpha_i_old = alpha[i];
        double alpha_j_old = alpha[j];
        double y_i = y[i];
        double y_j = y[j];
        double E_i = E[i];
        double E_j = E[j];
        
        double L = 0.0, H = 0.0;
        if (y_i != y_j) {
            L = std::max(0.0, alpha_j_old - alpha_i_old);
            H = std::min(C, C + alpha_j_old - alpha_i_old);
        } else {
            L = std::max(0.0, alpha_i_old + alpha_j_old - C);
            H = std::min(C, alpha_i_old + alpha_j_old);
        }
        if (L >= H) return false;
        
        double k_ii = K[i * N + i];
        double k_jj = K[j * N + j];
        double k_ij = K[i * N + j];
        double eta = 2.0 * k_ij - k_ii - k_jj;
        
        double alpha_j_new = 0.0;
        if (eta < 0.0) {
            alpha_j_new = alpha_j_old - y_j * (E_i - E_j) / eta;
            if (alpha_j_new < L) alpha_j_new = L;
            else if (alpha_j_new > H) alpha_j_new = H;
        } else {
            // Objective function evaluations
            double f1 = y_i * (E_i + b) - alpha_i_old * k_ii - y_i * y_j * alpha_j_old * k_ij;
            double f2 = y_j * (E_j + b) - y_i * y_j * alpha_i_old * k_ij - alpha_j_old * k_jj;
            double L1 = alpha_i_old + y_i * y_j * (alpha_j_old - L);
            double H1 = alpha_i_old + y_i * y_j * (alpha_j_old - H);
            
            double obj_L = L1 * f1 + L * f2 + 0.5 * L1 * L1 * k_ii + 0.5 * L * L * k_jj + y_i * y_j * L * L1 * k_ij;
            double obj_H = H1 * f1 + H * f2 + 0.5 * H1 * H1 * k_ii + 0.5 * H * H * k_jj + y_i * y_j * H * H1 * k_ij;
            
            if (obj_L < obj_H - 1e-7) alpha_j_new = L;
            else if (obj_L > obj_H + 1e-7) alpha_j_new = H;
            else alpha_j_new = alpha_j_old;
        }
        
        if (std::abs(alpha_j_new - alpha_j_old) < 1e-5 * (alpha_j_new + alpha_j_old + 1e-5)) {
            return false;
        }
        
        double alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new);
        if (alpha_i_new < 0.0) {
            alpha_j_new += y_i * y_j * alpha_i_new;
            alpha_i_new = 0.0;
        } else if (alpha_i_new > C) {
            alpha_j_new += y_i * y_j * (alpha_i_new - C);
            alpha_i_new = C;
        }
        
        double b_old = b;
        double b1 = b - E_i - y_i * (alpha_i_new - alpha_i_old) * k_ii - y_j * (alpha_j_new - alpha_j_old) * k_ij;
        double b2 = b - E_j - y_i * (alpha_i_new - alpha_i_old) * k_ij - y_j * (alpha_j_new - alpha_j_old) * k_jj;
        
        if (alpha_i_new > 0.0 && alpha_i_new < C) b = b1;
        else if (alpha_j_new > 0.0 && alpha_j_new < C) b = b2;
        else b = (b1 + b2) / 2.0;
        
        alpha[i] = alpha_i_new;
        alpha[j] = alpha_j_new;
        
        // Update error cache
        for (int k = 0; k < N; ++k) {
            E[k] = E[k] + y_i * (alpha_i_new - alpha_i_old) * K[i * N + k] +
                   y_j * (alpha_j_new - alpha_j_old) * K[j * N + k] + (b - b_old);
        }
        
        return true;
    };
    
    auto examineExample = [&](int i) -> bool {
        double y_i = y[i];
        double E_i = E[i];
        double r_i = y_i * E_i;
        
        if ((r_i < -eps && alpha[i] < C) || (r_i > eps && alpha[i] > 0.0)) {
            // Inner Loop 1: search non-bound
            int j = -1;
            double max_diff = -1.0;
            for (int k = 0; k < N; ++k) {
                if (alpha[k] > 0.0 && alpha[k] < C) {
                    double diff = std::abs(E_i - E[k]);
                    if (diff > max_diff) {
                        max_diff = diff;
                        j = k;
                    }
                }
            }
            if (j != -1 && takeStep(i, j)) return true;
            
            // Inner Loop 2: search non-bound from random start
            int start_idx = std::rand() % N;
            for (int k = 0; k < N; ++k) {
                int idx = (start_idx + k) % N;
                if (alpha[idx] > 0.0 && alpha[idx] < C) {
                    if (takeStep(i, idx)) return true;
                }
            }
            
            // Inner Loop 3: search entire dataset from random start
            start_idx = std::rand() % N;
            for (int k = 0; k < N; ++k) {
                int idx = (start_idx + k) % N;
                if (takeStep(i, idx)) return true;
            }
        }
        return false;
    };
    
    // 4. SMO Outer Loop
    int numChanged = 0;
    bool examineAll = true;
    int iter = 0;
    while ((numChanged > 0 || examineAll) && iter < max_iter) {
        numChanged = 0;
        if (examineAll) {
            for (int i = 0; i < N; ++i) {
                if (examineExample(i)) {
                    numChanged++;
                }
            }
        } else {
            for (int i = 0; i < N; ++i) {
                if (alpha[i] > 0.0 && alpha[i] < C) {
                    if (examineExample(i)) {
                        numChanged++;
                    }
                }
            }
        }
        
        if (examineAll) {
            examineAll = false;
        } else if (numChanged == 0) {
            examineAll = true;
        }
        iter++;
    }
    
    // 5. Identify support vectors
    int l = 0;
    for (int i = 0; i < N; ++i) {
        if (alpha[i] > 1e-7) {
            l++;
        }
    }
    
    // If no support vectors are found (extremely rare/impossible under normal optimization),
    // force at least one support vector to avoid allocating zero-size array
    if (l == 0) {
        l = 1;
    }
    
    svm_model* model = new svm_model();
    model->kernel_type = kernel_type;
    model->degree = degree;
    model->gamma = gamma;
    model->coef0 = coef0;
    model->l = l;
    model->D = D;
    model->SVs = new double[l * D];
    model->sv_coef = new double[l];
    model->rho = -b;
    model->probA = 0.0;
    model->probB = 0.0;
    
    int sv_idx = 0;
    for (int i = 0; i < N; ++i) {
        if (alpha[i] > 1e-7 || (sv_idx == 0 && i == N - 1 && l == 1)) {
            std::memcpy(&model->SVs[sv_idx * D], &X[i * D], D * sizeof(double));
            model->sv_coef[sv_idx] = alpha[i] * y[i];
            sv_idx++;
            if (sv_idx >= l) break;
        }
    }
    
    // 6. Platt scaling calibration
    if (probability) {
        std::vector<double> dec_values(N);
        for (int i = 0; i < N; ++i) {
            dec_values[i] = E[i] + y[i];
        }
        fit_platt_scaling(dec_values, y, N, model->probA, model->probB);
    }
    
    return model;
}

double svm_predict_decision(const svm_model* model, const double* x) {
    double sum = 0.0;
    for (int i = 0; i < model->l; ++i) {
        double k_val = kernel_function(&model->SVs[i * model->D], x, model->D, 
                                       model->kernel_type, model->gamma, model->coef0, model->degree);
        sum += model->sv_coef[i] * k_val;
    }
    return sum - model->rho;
}

double svm_predict(const svm_model* model, const double* x) {
    double dec = svm_predict_decision(model, x);
    return (dec >= 0.0) ? 1.0 : -1.0;
}

void svm_free_model(svm_model* model) {
    if (model) {
        delete[] model->SVs;
        delete[] model->sv_coef;
        delete model;
    }
}
