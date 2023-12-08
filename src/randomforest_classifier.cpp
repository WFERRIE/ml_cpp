#include "NumCpp.hpp"
#include <iostream>
#include "../include/randomforest_classifier.hpp"
#include "../include/rf_node.hpp"
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

double randomforest_classifier::compute_information_gain(nc::NdArray<double>& lc_y_bootstrap, nc::NdArray<double>& rc_y_bootstrap) {
    /*
    Compute information gain

    Example
    ---------
    double split_info_gain = compute_information_gain(left_child.y_bootstrap, right_child.y_bootstrap);    
    */

    double ig = 0.0;

    double p_parent = 0.0;
    double p_left = 0.0;
    double p_right = 0.0;

    nc::NdArray<double> parent_y_bootstrap = nc::append(lc_y_bootstrap, rc_y_bootstrap, nc::Axis::ROW);

    int n_samples_left = lc_y_bootstrap.shape().rows;
    int n_samples_right = rc_y_bootstrap.shape().rows;
    int n_samples_parent = n_samples_left + n_samples_right;

    if (n_samples_parent > 0) {
        int parent_class_0_count = nc::sum<int>(nc::where(parent_y_bootstrap == 0.0, 1, 0))(0, 0);
        p_parent = (double)parent_class_0_count / (double)n_samples_parent; // this is WRONG, should be # of times class 0 appears
    }

    if (n_samples_left > 0) {
        int left_class_0_count = nc::sum<int>(nc::where(lc_y_bootstrap == 0.0, 1, 0))(0, 0);
        p_left = (double)left_class_0_count / (double)n_samples_left;
    }

    if (n_samples_right > 0) {
        int right_class_0_count = nc::sum<int>(nc::where(rc_y_bootstrap == 0.0, 1, 0))(0, 0);
        p_right = (double)right_class_0_count / (double)n_samples_right;
    }

    double ig_parent = compute_entropy(p_parent);
    double ig_left = compute_entropy(p_left) * (double)n_samples_left / (double)n_samples_parent;
    double ig_right = compute_entropy(p_right) * (double)n_samples_right / (double)n_samples_parent;

    ig = ig_parent - ig_left - ig_right;

    return ig;

}

std::tuple<nc::NdArray<double>, nc::NdArray<double>, nc::NdArray<double>, nc::NdArray<double>> randomforest_classifier::bootstrap(nc::NdArray<double>& X, nc::NdArray<double>& y) {
    /*
    Perform bootstrapping operation. Bootstrapping is where you samples n samples WITH replacement. In this case, n is the number of samples
    in the dataset being passed. Function returns the bootstrapped set, as well as the out-of-bag set (out-of-bag, or oob is the set of samples
    not selected during bootstrapping which can be used as a pseudo-test-set).

    Parameters
    ----------
    X: Input data on which to perform bootstrapping. n samples will be bootstrapped where n is the number of samples in X.
    y: Labels on which to perform bootstrapping. n samples will be bootstrapped where n is the number of samples in X.

    Returns
    ----------
    Returns a tuple containing the following:
    [
        X_bootstrapped: Dataset of samples bootstrapped from X. Size (n_samples, n_features).
        y_bootstrapped: Dataset of samples bootstrapped from y. Size (n_samples, 1).
        X_oob: Dataset of samples NOT bootstrapped from X. Size (oob_size, n_features).
        y_oob: Dataset of samples NOT bootstrapped from y. Size (oob_size, n_features).
    ]

    Note that oob_size in theory is anywhere in the range [0, n_samples - 1], however, typically is approximately equal to n_samples / 3.

    Example
    ----------
    nc::NdArray<double> a1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    nc::NdArray<double> b1 = {-1, -2, -3};

    auto [a1_bootstrapped, b1_bootstrapped, a1_oob, b1_oob] = bootstrap(a1, b1);
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

rf_node randomforest_classifier::find_split(nc::NdArray<double>& X_bootstrap, nc::NdArray<double>& y_bootstrap, int max_features) {

    std::set<int> feature_list; // set to keep track of which samples have NOT been sampled
    int n_features = X_bootstrap.shape().cols;
    int n_samples = X_bootstrap.shape().rows;

    while (feature_list.size() < max_features) {
        int i = nc::random::randInt(0, n_features);
        feature_list.insert(i);
    }


    double best_info_gain = -999999.9;
    rf_node node = rf_node(0);

    for (auto& feat_idx: feature_list) { // for each feature
        // for each data point in that feature
        for (int i = 0; i < n_samples; i++) {
            double split_point = X_bootstrap(i, feat_idx); // this is the value to try splitting at
            rf_node left_child = rf_node(0);
            rf_node right_child = rf_node(0); // initialize two nodes to store the bootstraped data post-split

            for (int j = 0; j < n_samples; j++) {
                
                double value = X_bootstrap(j, feat_idx);
                nc::NdArray<double> X_row = X_bootstrap(j, X_bootstrap.cSlice());
                nc::NdArray<double> y_row = y_bootstrap(j, y_bootstrap.cSlice());

                if (value <= split_point) {
                    left_child.X_bootstrap = nc::append(left_child.X_bootstrap, X_row, nc::Axis::ROW);
                    left_child.y_bootstrap = nc::append(left_child.y_bootstrap, y_row, nc::Axis::ROW);
                }

                else {
                    right_child.X_bootstrap = nc::append(left_child.X_bootstrap, X_row, nc::Axis::ROW);
                    right_child.y_bootstrap = nc::append(left_child.y_bootstrap, y_row, nc::Axis::ROW);

                }

            }

            double split_info_gain = compute_information_gain(left_child.y_bootstrap, right_child.y_bootstrap);

            if (split_info_gain > best_info_gain) {
                // update details of best node to split at

                node.set_leftchild(&left_child);
                node.set_rightchild(&right_child);
                node.split_point = split_point;
                node.feature_idx = feat_idx;
                node.information_gain = split_info_gain;

                best_info_gain = split_info_gain;
            }

        }
        

    }


    return node;
    
}

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


