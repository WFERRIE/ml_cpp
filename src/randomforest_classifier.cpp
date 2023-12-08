#include "NumCpp.hpp"
#include <iostream>
#include "../include/randomforest_classifier.hpp"
#include <tuple>
#include <set>

// // int n_estimators;
// // int max_features;
// // int max_depth;
// // int min_samples_split;

double randomforest_classifier::compute_entropy(double p) {
    /* 
    Compute the entropy of some probability. 
    */
    double entropy;
    if (p == 0.0 || p == 1.0) {
        entropy = 0.0;
    }
    else {
        entropy = -(p * nc::log2(p) + (1.0 - p) * nc::log2(1 - p));
    }
    return entropy;

}

// void randomforest_classifier::compute_information_gain() {
//     /*
//     Compute information gain
//     */

    
// }

std::tuple<nc::NdArray<double>, nc::NdArray<double>, nc::NdArray<double>, nc::NdArray<double>> randomforest_classifier::bootstrap(nc::NdArray<double>& X, nc::NdArray<double>& y) {
    /*
    Perform bootstrapping operation. Bootstrapping is where you samples n samples WITH replacement. In this case, n is the number of samples
    in the dataset being passed.

    Parameters
    ----------
    X: Input data on which to perform bootstrapping. n samples will be bootstrapped where n is the number of samples in X.
    y: Labels on which to perform bootstrapping. n samples will be bootstrapped where n is the number of samples in X.

    Returns
    ----------
    Returns a tuple containing the bootstrapped data from X, and the bootstrapped data from y.

    Example
    ----------
    nc::NdArray<double> a1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    nc::NdArray<double> b1 = {-1, -2, -3};

    auto [a1_bootstrapped, b1_bootstrapped] = bootstrap(a1, b1);
    */

    y.reshape(-1, 1);

    int n_samples = X.shape().rows;
    int n_features = X.shape().cols;

    auto X_bootstrapped = nc::NdArray<double>(n_samples, n_features) = 0.0;
    auto y_bootstrapped = nc::NdArray<double>(n_samples, 1) = 0.0;



    std::set<int> oob_indices; // set to keep track of which samples have NOT been sampled

    for (int i = 0; i < n_samples; i++) {
        // initially, fill will all indices
        oob_indices.insert(i);
    }

    for (int i = 0; i < n_samples; i++) {
        // pick a random index between 0 (incl) and n_samples (excl)
        int j = nc::random::randInt(0, n_samples);


        // put the samples at that index into bootstrapped dataset
        nc::NdArray<double> X_slice = X(j, X.cSlice());
        nc::NdArray<double> y_slice = y(j, y.cSlice());
        X_bootstrapped.put(i, X_bootstrapped.cSlice(), X_slice);
        y_bootstrapped.put(i, y_bootstrapped.cSlice(), y_slice);

        oob_indices.erase(j); // erase index of sampled data (if j doesnt exist in oob_indices, nothing happens)
    }


    int n_oob_samples = oob_indices.size();

    auto X_oob = nc::NdArray<double>(n_oob_samples, n_features) = 0.0;
    auto y_oob = nc::NdArray<double>(n_oob_samples, 1) = 0.0;


    int oob_indices_idx = 0;
    for (auto& j: oob_indices) {   

        nc::NdArray<double> X_slice = X(j, X.cSlice());
        nc::NdArray<double> y_slice = y(j, y.cSlice());
        X_oob.put(oob_indices_idx, X_oob.cSlice(), X_slice);
        y_oob.put(oob_indices_idx, y_oob.cSlice(), y_slice);

        oob_indices_idx += 1;
    }

    return {X_bootstrapped, y_bootstrapped, X_oob, y_oob};

}

// void randomforest_classifier::compute_oob_score() {
//     //
// }

// void randomforest_classifier::find_split() {
//     //
// }

// void randomforest_classifier::terminal_node() {
//     //
// }

// void randomforest_classifier::split_node() {
//     //
// }

// void randomforest_classifier::build_tree() {
//     //
// }

// void randomforest_classifier::predict_tree() {
//     //
// }

// void randomforest_classifier::predict_rf() {
//     //
// }





randomforest_classifier::randomforest_classifier() {

}

randomforest_classifier::~randomforest_classifier() {

}

// void randomforest_classifier::fit(nc::NdArray<double>& X, nc::NdArray<double>& y, bool verbose) {
//     //
// }

// nc::NdArray<double> randomforest_classifier::predict(nc::NdArray<double>& X) {

// } 


