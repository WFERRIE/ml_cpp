#include "NumCpp.hpp"
#include <iostream>
#include "logistic_regression.h"


double logistic_regression::compute_BCE_cost(nc::NdArray<double> predictions, nc::NdArray<double> y) {

    auto n_samples = predictions.shape().rows;

    auto cost = -(1.0 / n_samples) * nc::sum<double>(y * nc::log(predictions) + (1.0 - y) * nc::log(1.0 - predictions)); // cross entropy loss

    return cost(0, 0); // return it as a double instead of a 1 element nc::NdArray
}

nc::NdArray<double> logistic_regression::sigmoid(nc::NdArray<double> z) {
    return 1.0 / (1.0 + nc::exp(-z));
}


logistic_regression::logistic_regression(const int& n_iters, const double& lr) : n_iters(n_iters), lr(lr) {
    // constructor
}

logistic_regression::~logistic_regression() {
    // destructor
}

const nc::NdArray<double>& logistic_regression::get_weights() const {
    return weights;
}

const double& logistic_regression::get_bias() const {
    return bias;
}


void logistic_regression::fit(nc::NdArray<double> X, nc::NdArray<double> y, bool verbose) {

    int n_samples = X.shape().rows;
    int n_features = X.shape().cols;

    weights = nc::zeros<double>(n_features, 1);

    nc::NdArray<double> predictions;

    for (int i = 0; i < n_iters; i++) {
        auto z = nc::dot<double>(X, weights) + bias;
        predictions = sigmoid(z);

        auto dw = 1.0 / n_samples * nc::dot(X.transpose(), (predictions - y));
        auto db = 1.0 / n_samples * nc::sum<double>(predictions - y);


        weights = weights - lr * dw;
        bias = bias - (lr * db)(0, 0); // convert 1 element nc::NdArray to a double by accessing the (0, 0) index    
        if (verbose == true) {
            std::cout << "Cost at iteration " << i << ": " << compute_BCE_cost(predictions, y) << std::endl;
        }
    }

}

nc::NdArray<double> logistic_regression::predict(nc::NdArray<double> X) {
    int n_samples = X.shape().rows;
    auto z = nc::dot<double>(X, weights) + bias;
    nc::NdArray<double> predictions = nc::round(sigmoid(z));

    return predictions;
}
