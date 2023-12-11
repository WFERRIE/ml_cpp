#include "NumCpp.hpp"
#include <iostream>
#include "../include/randomforest_classifier.hpp"
#include "../include/rf_node.hpp"
#include <tuple>
#include <set>
#include <vector>
#include "../include/utils.hpp"
#include <math.h>

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

double randomforest_classifier::compute_oob_score(rf_node* tree, nc::NdArray<double>& X_test, nc::NdArray<double>& y_test) {
    // copmpute the decision tree's score on the out of bag set
    int correct_labels = 0;
    int n_samples = X_test.shape().rows;
    for (int i = 0; i < n_samples; i++) {
        double prediction = predict_tree(tree, X_test(i, X_test.cSlice()));
        if (prediction == y_test(i, y_test.cSlice())) {
            correct_labels += 1;
        }
    }

    return (double)correct_labels / (double)n_samples;
}


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


double randomforest_classifier::calculate_terminal_node(rf_node* node) {
    auto y_bootstrap = (*node).y_bootstrap;
    return nc::mean(y_bootstrap)(0, 0);
}


rf_node randomforest_classifier::split_node(rf_node* node, int max_features, int min_samples_split, int max_depth, int depth) {
    rf_node* left_child = node->get_leftchild();
    rf_node* right_child = node->get_rightchild();

    int left_n_samples = (left_child->y_bootstrap).shape().rows;
    int right_n_samples = (right_child->y_bootstrap).shape().rows;

    if (left_n_samples == 0 || right_n_samples == 0) {
        // if one of our children has no samples left, set the leaves of the tree and return the tree
        
        nc::NdArray<double> combined_y_boosted = nc::append(left_child->y_bootstrap, right_child->y_bootstrap, nc::Axis::ROW);
        
        rf_node terminal_node = rf_node(0);
        terminal_node.y_bootstrap = combined_y_boosted; 

        terminal_node.is_leaf = true;
        terminal_node.leaf_value = calculate_terminal_node(&terminal_node);

        node->set_leftchild(&terminal_node);
        node->set_rightchild(&terminal_node);
        
        return *node;
    }

    else if (depth >= max_depth) {
        // if we have hit the max depth, make the children leaves
        (*left_child).leaf_value = calculate_terminal_node(left_child);
        (*left_child).is_leaf = true;

        (*right_child).leaf_value = calculate_terminal_node(right_child);
        (*right_child).is_leaf = true;


        // note: double check the code above actually modifies node.left_child and node.right_child
        return *node;
    }

    if (left_child->X_bootstrap.shape().rows <= min_samples_split) {
        // if the left child has less samples than our min_samples_split, set the left child to be a leaf
        (*left_child).leaf_value = calculate_terminal_node(left_child);
        (*left_child).is_leaf = true;
    }

    else {
        // otherwise, split the left child further
        rf_node split_left_child = find_split(left_child->X_bootstrap, left_child->y_bootstrap, max_features);
        node->set_leftchild(&split_left_child);
        // recursively continue splitting the left child further
        split_node(node->get_leftchild(), max_features, min_samples_split, max_depth, depth + 1);
    }

    if (right_child->X_bootstrap.shape().rows <= min_samples_split) {
        // if the left child has less samples than our min_samples_split, set the left child to be a leaf
        (*right_child).leaf_value = calculate_terminal_node(right_child);
        (*right_child).is_leaf = true;
    }

    else {
        // otherwise, split the left child further
        rf_node split_right_child = find_split(right_child->X_bootstrap, right_child->y_bootstrap, max_features);
        node->set_leftchild(&split_right_child);
        // recursively continue splitting the left child further
        split_node(node->get_rightchild(), max_features, min_samples_split, max_depth, depth + 1);
    }
    
    return *node; // should return the root node

}



rf_node randomforest_classifier::build_tree(nc::NdArray<double>& X_bootstrap, nc::NdArray<double>& y_bootstrap) {
    // begin tree building process

    rf_node root = find_split(X_bootstrap, y_bootstrap, max_features);
    split_node(&root, max_features, min_samples_split, max_depth, 1);

    return root;
}


double randomforest_classifier::predict_tree(rf_node* tree, nc::NdArray<double> X_sample) {
    /*
    use tree to predict label

    Parameters
    ----------
    tree: rf_node that represents the root of the tree
    X_sample: nc::NdArray<double> of size (1, n_features) to predict on.

    Returns
    ---------
    leaf_value: the class predicted by the tree on sample X_sample
    */ 

    X_sample.reshape(1, -1); // ensure sample is in row format

    int feature_idx = (*tree).feature_idx; // index of feature to decide on

    if (X_sample(0, feature_idx) <= (*tree).split_point) { // check if we want to go to the left or right child
        // if our value is less than the split point
        if (tree->is_leaf) {
            return tree->leaf_value;
        }
        else {
            return predict_tree(tree->get_leftchild(), X_sample);
        }
    }
    else {
        // if our value is greater than the split point
        if (tree->is_leaf) {
            return tree->leaf_value;
        }
        else {
            return predict_tree(tree->get_rightchild(), X_sample);
        }
    }
}


randomforest_classifier::randomforest_classifier(const int n_estimators, const int max_depth, const int min_samples_split, int max_features) : n_estimators(n_estimators), max_depth(max_depth), min_samples_split(min_samples_split), max_features(max_features) {

}

randomforest_classifier::~randomforest_classifier() {

}

void randomforest_classifier::fit(nc::NdArray<double>& X_train, nc::NdArray<double>& y_train, bool verbose) {
    // main fitting function to be called by user

    if (max_features == -1) {
        // this is the default value, indicating the user didn't specify a max # feats
        // by default we will use the sqrt of the number of features

        int n_features = X_train.shape().cols;
        max_features = sqrt(n_features);
    }

    for (int i = 0; i < n_estimators; i++) {
        auto [X_bootstrapped, y_bootstrapped, X_oob, y_oob] = bootstrap(X_train, y_train);
        rf_node tree = build_tree(X_bootstrapped, y_bootstrapped);
        tree_list.push_back(&tree);
        double oob_score = compute_oob_score(&tree, X_oob, y_oob);
        oob_list.push_back(oob_score);
    }
}


nc::NdArray<double> randomforest_classifier::predict(nc::NdArray<double>& X) {

    nc::NdArray<double> predictions;

    int n_samples = X.shape().rows;
    int n_trees = tree_list.size();

    for (int i = 0; i < n_samples; i++) {
        std::vector<double> ensemble_results;
        for (int j = 0; j < n_trees; j++) {
            rf_node* tree_ptr = tree_list[j];
            double pred = predict_tree(tree_ptr, X(i, X.cSlice()));
            ensemble_results.push_back(pred);
        }

        // get most frequent value from the ensemble_results
        nc::NdArray<double> f = {get_most_frequent_element(ensemble_results)};
        predictions = nc::append(predictions, f, nc::Axis::ROW);
        
    }

    return predictions;
}

