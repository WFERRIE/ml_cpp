#include "NumCpp.hpp"
#include <iostream>
#include "../include/logistic_regression.hpp"
#include <string>


double logistic_regression::compute_BCE_cost(nc::NdArray<double>& predictions, nc::NdArray<double>& y) {

    auto n_samples = predictions.shape().rows;

    double reg_term;

    auto cost = -(1.0 / n_samples) * nc::sum<double>(y * nc::log(predictions) + (1.0 - y) * nc::log(1.0 - predictions)); // cross entropy loss

    if (penalty == "l1") {
        reg_term = reg_strength * nc::sum(nc::abs(weights))(0, 0);
    }

    else if (penalty == "l2") {
        reg_term = reg_strength * nc::sum(nc::power<double>(weights, 2))(0, 0);
    }

    else {
        reg_term = 0.0;
    }

    return cost(0, 0) + reg_term; // return it as a double instead of a 1 element nc::NdArray
}


nc::NdArray<double> logistic_regression::sigmoid(nc::NdArray<double>& z) {
    return 1.0 / (1.0 + nc::exp(-z));
}


logistic_regression::logistic_regression(const std::string penalty, const double reg_strength, const int max_iters, const double lr, const double tol, const int init_mode) : penalty(penalty), reg_strength(reg_strength), max_iters(max_iters), lr(lr), tol(tol), init_mode(init_mode) {
    // constructor
    if (lr <= 0) {
        std::runtime_error("Learning rate must be greater than 0.");
    }

    if (max_iters <= 0) {
        std::runtime_error("Number of iterations must be greater than 0.");
    }

}

logistic_regression::~logistic_regression() {
    // destructor
}

const nc::NdArray<double> logistic_regression::get_weights() const {
    if (!is_fit) {
        std::runtime_error("Error: Please call .fit() function before attempting to call any getters.");
    }
    return weights;
}

const nc::NdArray<double> logistic_regression::get_bias() const {
    if (!is_fit) {
        std::runtime_error("Error: Please call .fit() function before attempting to call any getters.");
    }
    return bias;
}


void logistic_regression::fit(nc::NdArray<double>& X, nc::NdArray<double>& y, bool verbose) {

    is_fit = true;

    nc::uint32 n_samples = X.shape().rows;
    nc::uint32 n_features = X.shape().cols;

    n_classes = nc::unique(y).shape().cols;

    double curr_cost;
    double prev_cost = 9999999; // set previous cost to some very high number

    if (init_mode == 1) {

        weights = nc::random::rand<double>({n_features, n_classes}) / 100.0; // initialize weights with small random perturbations around 0
    }
    
    else {
        weights = nc::zeros<double>(n_features, n_classes); // initialize with all weights of 0
    }

    bias = nc::zeros<double>(1, n_classes);
    
    nc::NdArray<double> predictions;

    for (double class_idx = 0.0; class_idx < n_classes; class_idx++) {

        nc::NdArray<double> temp_weights = weights(weights.rSlice(), class_idx); // temp_weights are the weights for the class we are currently working with
        double temp_bias = bias(bias.rSlice(), class_idx)(0, 0); // temp_bias is the bias for the class we are currently working with

        nc::NdArray<double> binary_labels = nc::where(y == class_idx, 1.0, 0.0); // mask labels for the current class

        for (int i = 0; i < max_iters; i++) { // training loop

            auto z = nc::dot<double>(X, temp_weights) + temp_bias;
            predictions = sigmoid(z);

            auto dw = 1.0 / n_samples * nc::dot(X.transpose(), (predictions - binary_labels));
            auto db = 1.0 / n_samples * nc::sum<double>(predictions - binary_labels);

            if (penalty == "l1") {
                dw += (reg_strength / (double)n_samples) * nc::where(temp_weights >= 0.0, 1.0, -1.0);
            }

            else if (penalty == "l2") {
                dw += (reg_strength / (double)n_samples) * 2.0 * temp_weights;
            }

            temp_weights = temp_weights - lr * dw;
            temp_bias = temp_bias - (lr * db)(0, 0); // convert 1 element nc::NdArray to a double by accessing the (0, 0) index  

            if (i % 100 == 0) { 
                curr_cost = compute_BCE_cost(predictions, binary_labels);

                if (nc::abs(prev_cost - curr_cost) < tol) {
                    if (verbose) {
                        std::cout << "Stopping training early. Training has converged on iteration:" << i << std::endl;
                        std::cout << "Final Cost:" << curr_cost << std::endl;
                    }
                    
                    break;
                }

                if (verbose) {
                    std::cout << "Cost at iteration " << i << ": " << compute_BCE_cost(predictions, binary_labels) << std::endl;
                }
            }

            prev_cost = curr_cost; // update previous cost

        }

        weights.put(weights.rSlice(), class_idx, temp_weights); //update weights for class that was just trained
        bias.put(bias.rSlice(), class_idx, temp_bias);
    }



}

nc::NdArray<double> logistic_regression::predict(nc::NdArray<double>& X) {

    int n_samples = X.shape().rows;

    auto z = nc::dot<double>(X, weights) + bias;
    nc::NdArray<double> predictions = sigmoid(z);

    nc::NdArray<nc::uint32> predictions_out_int = nc::argmax(predictions, nc::Axis::COL).transpose();

    auto predictions_out = predictions_out_int.astype<double>();

    return predictions_out;
}
