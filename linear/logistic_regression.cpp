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


logistic_regression::logistic_regression(const int& n_iters, const double& lr, const int& init_mode) : n_iters(n_iters), lr(lr), init_mode(init_mode){
    // constructor
}

logistic_regression::~logistic_regression() {
    // destructor
}

const nc::NdArray<double>& logistic_regression::get_weights() const {
    return weights;
}

const nc::NdArray<double>& logistic_regression::get_bias() const {
    return bias;
}


void logistic_regression::fit(nc::NdArray<double> X, nc::NdArray<double> y, bool verbose) {

    nc::uint32 n_samples = X.shape().rows;
    nc::uint32 n_features = X.shape().cols;

    n_classes = nc::unique(y).shape().cols;

    if (init_mode == 1) {

        weights = nc::random::rand<double>({n_features, n_classes}) / 100.0; // initialize weights with small random perturbations around 0

    }
    else {
        weights = nc::zeros<double>(n_features, n_classes); // initialize with all weights of 0
    }

    bias = nc::zeros<double>(1, n_classes);
    
    nc::NdArray<double> predictions;

    for (double class_idx = 0.0; class_idx < n_classes; class_idx++) {

        nc::NdArray<double> temp_weights = weights(weights.rSlice(), class_idx);
        double temp_bias = bias(bias.rSlice(), class_idx)(0, 0);

        nc::NdArray<double> binary_labels = nc::where(y == class_idx, 1.0, 0.0); // mask labels for the current class

        for (int i = 0; i < n_iters; i++) { // training loop

            auto z = nc::dot<double>(X, temp_weights) + temp_bias;
            predictions = sigmoid(z);

            auto dw = 1.0 / n_samples * nc::dot(X.transpose(), (predictions - binary_labels));
            auto db = 1.0 / n_samples * nc::sum<double>(predictions - binary_labels);


            temp_weights = temp_weights - lr * dw;
            temp_bias = temp_bias - (lr * db)(0, 0); // convert 1 element nc::NdArray to a double by accessing the (0, 0) index    
            if (verbose == true) {
                std::cout << "Cost at iteration " << i << ": " << compute_BCE_cost(predictions, binary_labels) << std::endl;
            }
        }

        weights.put(weights.rSlice(), class_idx, temp_weights); //update weights for class that was just trained
        bias.put(bias.rSlice(), class_idx, temp_bias);
    }

}

nc::NdArray<double> logistic_regression::predict(nc::NdArray<double> X) {
    int n_samples = X.shape().rows;

    auto z = nc::dot<double>(X, weights) + bias;
    nc::NdArray<double> predictions = sigmoid(z);

    nc::NdArray<nc::uint32> predictions_out_int = nc::argmax(predictions, nc::Axis::COL).transpose();

    auto predictions_out = predictions_out_int.astype<double>();

    return predictions_out;
}
