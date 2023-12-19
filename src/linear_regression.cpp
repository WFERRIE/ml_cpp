#include "NumCpp.hpp"
#include <iostream>
#include "../include/linear_regression.hpp"
#include <string>

linear_regression::linear_regression(const std::string penalty, const double reg_strength, const int max_iters, const double lr, const double tol, const int init_mode) : penalty(penalty), reg_strength(reg_strength), max_iters(max_iters), lr(lr), tol(tol), init_mode(init_mode) {
    /*
    Constructor for the model.

    Parameters
    ---------
    penalty: string represnting the type of regularization to perform. "l1" will result in lasso regression, and "l2" will result in ridge regression. Defaults to "l2".

    reg_strength: double representing the strength of regularization. Defaults to 0.1

    max_iters: integer represeting the maximum number of iterations before the model will stop training. Defaults to 1000.

    lr: double representing the learning rate, or how large the updates to the model weights are on each iteration. Default to 0.01

    tol: double representing the convergence criteria, or in other words, 
            the minimum change in cost on each iteration required for the model to continue training.

    init_mode: integer representing type of weights initialization. If set to 1, model weights will be initialized to small values around 0.
                This provides some initial randomness and may help with numerical stability.
    
    */

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
    // returns the weights of the model.
    return weights;
}


const double linear_regression::get_bias() const {
    // returns the bias of the model.
    return bias;
}


nc::NdArray<double> linear_regression::add_bias_feature(nc::NdArray<double>& X) {
    // appends a bias feature to the data X. Used during training, not meant to be
    // called by the user.

    auto X_with_bias = nc::hstack({nc::ones<double>(X.shape().rows, 1), X});

    return X_with_bias;    

}


double linear_regression::compute_cost(nc::NdArray<double>& X, nc::NdArray<double>& y_true) {
    /*
    Computes the cost of the model by predicting based on X and comparing to y_true.
    Used during training, not meant to be called by the user.
    */

    double reg_term;

    auto y_pred = predict(X, true);

    double cost = nc::mean(nc::power<double>(y_true - y_pred, 2.0))(0, 0);

    if (penalty == "l1") {
        reg_term = reg_strength * nc::sum(nc::abs(weights))(0, 0);
    }

    else if (penalty == "l2") {
        reg_term = reg_strength * nc::sum(nc::power<double>(weights, 2))(0, 0);
    }

    else {
        reg_term = 0.0;
    }

    return cost + reg_term;
}


nc::NdArray<double> linear_regression::calculate_gradient(nc::NdArray<double>& X, nc::NdArray<double>& y) {
    /*
    Calculates the gradient based on X and y. This function is called in the .fit() method and is used for training
    purposes. Not meant to be called by the user.
    */

    auto weights_sliced = weights(weights.rSlice(), {1, (int)weights.shape().cols}); // slice weights to not include the first feature, which is the bias feature

    int n_samples = X.shape().rows;

    auto y_pred = predict(X, true);
    
    auto db = -2.0 * nc::sum<double>(y - y_pred); // dJ/dw_0

    auto dx = -2.0 * nc::sum<double>(X(X.rSlice(), {1, (int)X.shape().cols}) * (y - y_pred).reshape(-1, 1), nc::Axis::ROW); // dJ/dw_i

    if (penalty == "l1") {
        dx += (reg_strength / (double)n_samples) * nc::where(weights_sliced >= 0.0, 1.0, -1.0);
    }

    else if (penalty == "l2") {
        dx += (reg_strength / (double)n_samples) * 2.0 * weights_sliced;
    }
        
    auto grad = nc::hstack({db, dx}); // gradient

    return grad / (double)X.shape().rows;
}


void linear_regression::fit(nc::NdArray<double>& X, nc::NdArray<double>& y, bool verbose) {
    /*
    Function to fit linear_regression model on input data X and input labels y.

    Parameters
    ---------
    X: nc::NdArray of shape (n_samples, n_features) on which to train model.
    y: nc::NdArray of shape (n_samples, 1) or (1, n_samples) containing ground truth labels.
    verbose: boolean - if true, model will print out the cost at each iteration.


    Returns
    ---------
    Does not return anything. However, does set the is_fit flag to true, allowing
    the user to call the .predict() method.


    Example
    ----------
    const std::string penalty = "l1";
    const double reg_strength = 1.0;
    const int max_iters = 1000000;
    const double lr = 0.01;
    const double tol = 0.00001;
    const int init_mode = 1;
    const verbose = true;
    linear_regression lin_reg = linear_regression(penalty, reg_strength, max_iters, lr, tol, init_mode);
    lin_reg.fit(X, y, verbose);
    nc::NdArray<double> y_pred = lin_reg.predict(X);
    */

    is_fit = true; // this needs to be set to true immediately otherwise the .predict() method which is used in this method will error.

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

        grad = calculate_gradient(X_with_bias, y); // calculate gradient

        weights -= lr * grad; // update weights

        curr_cost = compute_cost(X_with_bias, y); // get cost

        if (i % 100 == 0) {

            if (nc::abs(prev_cost - curr_cost) < tol) {
                std::cout << "Stopping training early. Training has converged on iteration:" << i << std::endl;
                std::cout << "Final Cost:" << curr_cost << std::endl;
                break;
            }
        }

        prev_cost = curr_cost; // update previous cost

        if (verbose == true) {
            std::cout << "Cost at iteration " << i << ": " << curr_cost << std::endl;
        }
    }
}

nc::NdArray<double> linear_regression::predict(nc::NdArray<double>& X, bool bias_feature) {
    /*
    Function to predict an output based on input X.

    Parameters
    ---------
    X: nc::NdArray of shape (n_samples, n_features) on which to predict.
    bias_feature: boolean - set to true only if input data X already contains a bias feature. Defaults to false.


    Returns
    ---------
    y_pred: nc::NdArray<double> of shape (n_samples, 1) which contains the predicted values.


    Example
    ----------
    const std::string penalty = "l1";
    const double reg_strength = 1.0;
    const int max_iters = 1000000;
    const double lr = 0.01;
    const double tol = 0.00001;
    const int init_mode = 1;
    const verbose = true;
    linear_regression lin_reg = linear_regression(penalty, reg_strength, max_iters, lr, tol, init_mode);
    lin_reg.fit(X, y, verbose);
    nc::NdArray<double> y_pred = lin_reg.predict(X);
    */
    if (!is_fit) {
        std::runtime_error("Error: Model has not yet been fit. Please call .fit() method before attempting to call .predict() method.");
    }

    nc::NdArray<double> X_;

    if (!bias_feature) { // if there is no bias feature present
        X_ = add_bias_feature(X);
    }
    else { // if there is a bias feature present
        X_ = X;
    }
    auto y_pred = nc::dot<double>(X_, weights.transpose());
    return y_pred;

}
