#include "NumCpp.hpp"
#include <iostream>
#include "linear_regression.h"
#include <string>

linear_regression::linear_regression(const std::string penalty, const double reg_strength, const int max_iters, const double lr, const double tol, const int init_mode) : penalty(penalty), reg_strength(reg_strength), max_iters(max_iters), lr(lr), tol(tol), init_mode(init_mode) {
    // constructor
    if (lr <= 0) {
        throw std::runtime_error("Learning rate must be greater than 0.");
    }

    if (max_iters <= 0) {
        throw std::runtime_error("Number of iterations must be greater than 0.");
    }

}


linear_regression::~linear_regression() {
    // destructor
}


const nc::NdArray<double> linear_regression::get_weights() const {
    return weights;
}


const double linear_regression::get_bias() const {
    return bias;
}


nc::NdArray<double> linear_regression::add_bias_feature(nc::NdArray<double>& X) {

    auto X_with_bias = nc::hstack({nc::ones<double>(X.shape().rows, 1), X});

    return X_with_bias;    

}


double linear_regression::compute_cost(nc::NdArray<double>& X, nc::NdArray<double>& y_true) {

    double reg_term = 0.0;

    std::cout << "compute_cost() is calling predict()" << std::endl;

    auto y_pred = predict(X, true);

    std::cout << "compute_cost() is calculating the cost" << std::endl;

    std::cout << nc::power<double>(y_true - y_pred, 2.0).shape() << std::endl;

    std::cout << nc::mean(nc::power<double>(y_true - y_pred, 2.0)) << std::endl;

    double cost = nc::mean(nc::power<double>(y_true - y_pred, 2.0))(0, 0);

    std::cout << "cost: " << cost << std::endl;

    return cost + reg_term; // return it as a double instead of a 1 element nc::NdArray
}


nc::NdArray<double> linear_regression::calculate_gradient(nc::NdArray<double>& X, nc::NdArray<double>& y) {

    std::cout << "calculating gradient" << std::endl;
    auto y_pred = predict(X, true);
    
    std::cout << "calculating db" << std::endl;
    auto db = -2.0 * nc::sum<double>(y - y_pred); // dJ/dw_0

    std::cout << "calcing dx" << std::endl;

    auto dx = -2.0 * nc::sum<double>(X(X.rSlice(), {1, (int)X.shape().cols}) * (y - y_pred).reshape(-1, 1), nc::Axis::ROW); // dJ/dw_i

    std::cout << "calcing grad" << std::endl;
    auto grad = nc::hstack({db, dx}); // gradient

    std::cout << "calculate_gradient() is retruning the gradient" << std::endl;

    return grad / (double)X.shape().rows;
}


void linear_regression::fit(nc::NdArray<double>& X, nc::NdArray<double>& y, bool verbose) {

    auto X_with_bias = add_bias_feature(X);

    nc::uint32 n_samples = X_with_bias.shape().rows;
    nc::uint32 n_features = X_with_bias.shape().cols;

    double curr_cost;
    double prev_cost = 9999999; // set previous cost to some very high number

    nc::NdArray<double> y_pred; // predicted labels
    nc::NdArray<double> grad; // weights gradient

    if (init_mode == 1) {
        weights = nc::random::rand<double>({1, n_features}) / 100.0; // initialize weights with small random perturbations around 0

    }
    else {
        weights = nc::zeros<double>(1, n_features); // initialize with all weights of 0
    }
    
    for (int i = 0; i < max_iters; i++) { // training loop

        std::cout << "calcualting gradient" << std::endl;
        grad = calculate_gradient(X_with_bias, y); // calculate gradient
        std::cout << "gradient calculated" << std::endl;


        weights -= lr * grad; // update weights

        std::cout << "computing cost" << std::endl;

        curr_cost = compute_cost(X_with_bias, y); // get cost

        std::cout << "cost computed" << std::endl;

        if (i % 100 == 0) {

            if (nc::abs(prev_cost - curr_cost) < tol) {
                std::cout << "Stopping training early. Training has converged on iteration:" << i << std::endl;
                std::cout << "Final Cost:" << curr_cost << std::endl;
                break;
            }
        }

        std::cout << "updating prev cost" << std::endl;

        prev_cost = curr_cost; // update previous cost

        std::cout << "doing verbose thing" << std::endl;

        if (verbose == true) {
            std::cout << "Cost at iteration " << i << ": " << curr_cost << std::endl;
        }
    }
}

nc::NdArray<double> linear_regression::predict(nc::NdArray<double>& X, bool bias_feature) {
    nc::NdArray<double> X_;

    if (!bias_feature) { // if there is no bias feature present
        X_ = add_bias_feature(X);
    }
    else { // if there is a bias feature present
        X_ = X;
    }
    std::cout << "running prediction" << std::endl;
    return nc::dot<double>(X_, weights.transpose());

}
