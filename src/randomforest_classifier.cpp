#include "NumCpp.hpp"
#include <iostream>
#include "../include/randomforest_classifier.hpp"
#include "../include/rf_node.hpp"
#include <tuple>
#include <set>
#include <vector>
#include "../include/utils.hpp"
#include <math.h>
#include <random>


randomforest_classifier::randomforest_classifier(const int n_estimators, const int max_depth, const int min_samples_split, int max_features) : n_estimators(n_estimators), max_depth(max_depth), min_samples_split(min_samples_split), max_features(max_features) {
    /*
    Constructor for the random forest classifier model.

    Parameters
    ----------
    n_estimators: how many decision trees to build in the random forest. Defaults to 10.

    max_depth: maximum depth to which to grow each decision tree. Defaults to 5.

    min_samples_split: the minimum number of training samples at a node to continue growing the tree by splitting that node. Defaults to 5.

    max_features: maximum number of features to consider while growing each decision tree. If left at default value of -1,
                    max_features will be set to the square root of the total number of features in the training data, cast to an integer value.
    
    */
}

randomforest_classifier::~randomforest_classifier() {
    // Destructor. Calls the destructor on the root node of each tree in tree_list,
    // which in turn will recursively call the destructors on all children.
    if (is_fit) {
        for (auto t : tree_list) {
            delete t;
        }
    }
}


void randomforest_classifier::fit(nc::NdArray<double>& X_train, nc::NdArray<double>& y_train) {
    /*
    Main fitting function. To be called by the user.

    Parameters
    ----------
    X_train: nc::NdArray<double> of size (n_samples, n_features) contianing training data.
    y_train: nc::NdArray<double> of size (n_samples, 1) or (1, n_samples) containing ground truth labels.

    Returns
    ----------
    Nothing, however, sets is_fit flag to true, allowing user to then call the .predict() method.

    Example
    ----------
    int n_estimators = 4;
    int max_depth = 5;
    int min_samples_split = 10;
    int max_features = 5;

    randomforest_classifier rfc = randomforest_classifier(n_estimators, max_depth, min_samples_split, max_features);

    rfc.fit(X_train, y_train);

    auto y_pred = rfc.predict(X_test);
     
    */

    if (max_features == -1) {
        // this is the default value, indicating the user didn't specify a max # feats
        // by default we will use the sqrt of the number of features

        int n_features = X_train.shape().cols;
        max_features = sqrt(n_features);
    }

    else if (max_features > X_train.shape().cols) {
        std::runtime_error("Erorr: max_features is > total number of features in training data.");
    }

    else if (max_features <= 0) {
        std::runtime_error("Error: max_features is <= 0. Please ensure max_features is at least 1.");
    }

    for (int i = 0; i < n_estimators; i++) {
        int seed = rand();
        nc::random::seed(seed);

        rf_node* root = new rf_node();
        
        std::cout << "Fitting Estimator #" << i << std::endl;

        auto [X_bootstrapped, y_bootstrapped, X_oob, y_oob] = bootstrap(X_train, y_train);

        build_tree(root, X_bootstrapped, y_bootstrapped);

        tree_list.push_back(root);
        double oob_score = compute_oob_score(root, X_oob, y_oob);
        oob_list.push_back(oob_score);
    }

    is_fit = true;
}


nc::NdArray<double> randomforest_classifier::predict(nc::NdArray<double>& X) {
    /*
    Function to predict outputs based on input X.

    Parameters
    ----------
    X: nc::NdArray<double> of size (n_samples, n_features) containing input data on which to predict.

    Returns
    ----------
    y_pred: nc::NdArray<double> of size (n_samples, 1) containing predictions.

    Example
    ----------
    int n_estimators = 4;
    int max_depth = 5;
    int min_samples_split = 10;
    int max_features = 5;

    randomforest_classifier rfc = randomforest_classifier(n_estimators, max_depth, min_samples_split, max_features);

    rfc.fit(X_train, y_train);

    auto y_pred = rfc.predict(X_test);

    */
    if (!is_fit) {
        std::runtime_error("Error: Estimator has not been fit. Please call .fit() method before calling .predict() method.");
    }

    nc::NdArray<double> predictions;

    int n_samples = X.shape().rows;
    int n_trees = tree_list.size();

    for (int i = 0; i < n_samples; i++) {
        std::vector<double> ensemble_results;
        for (int j = 0; j < n_trees; j++) {
            rf_node* tree_ptr = tree_list[j];
            double pred = predict_sample(tree_ptr, X(i, X.cSlice()));
            ensemble_results.push_back(pred);
        }

        // get most frequent value from the ensemble_results
        nc::NdArray<double> f = { get_most_frequent_element(ensemble_results) };
        predictions = nc::append(predictions, f, nc::Axis::ROW);

    }

    return predictions;
}


void randomforest_classifier::build_tree(rf_node* root, nc::NdArray<double>& X_bootstrap, nc::NdArray<double>& y_bootstrap) {
    // begin tree building process. Bootstraps the data and then performs the splitting process on the root.
    root->X_bootstrap = X_bootstrap;
    root->y_bootstrap = y_bootstrap;

    find_split(root);
    split_node(root, max_features, min_samples_split, max_depth, 1);

    return;
}


void randomforest_classifier::find_split(rf_node* parent_node) {
    // finds the optimal point in the data on which to split the data. Saves
    // this information into the parent_node. This function must be called
    // on a node before split_node() may be called on that node.

    nc::NdArray<double> X_bootstrap = parent_node->X_bootstrap;
    nc::NdArray<double> y_bootstrap = parent_node->y_bootstrap;

    std::set<int> feature_list; // set to keep track of which samples have NOT been sampled
    int n_features = X_bootstrap.shape().cols;
    int n_samples = X_bootstrap.shape().rows;

    while (feature_list.size() < max_features) {
        // randomly sample features
        int i = nc::random::randInt(0, n_features);
        feature_list.insert(i);
    }


    double best_info_gain = -999999.9;
    
    rf_node* left_child = new rf_node(); // create two children nodes for the parent node
    rf_node* right_child = new rf_node();

    parent_node->set_leftchild(left_child); // set the children as children nodes for the parent
    parent_node->set_rightchild(right_child);

    for (auto& feat_idx: feature_list) { // for each feature
        
        for (int i = 0; i < n_samples; i++) { // for each data point in that feature

            rf_node temp_left_child = rf_node(); // temp nodes to split the bootstrapped data into
            rf_node temp_right_child = rf_node();

            double split_point = X_bootstrap(i, feat_idx); // this is the value to try splitting at

            for (int j = 0; j < n_samples; j++) {
                
                double value = X_bootstrap(j, feat_idx);
                nc::NdArray<double> X_row = X_bootstrap(j, X_bootstrap.cSlice());
                nc::NdArray<double> y_row = y_bootstrap(j, y_bootstrap.cSlice());

                if (value <= split_point) { // sort the data into the left or right child based on the split point
                    temp_left_child.X_bootstrap = nc::append(temp_left_child.X_bootstrap, X_row, nc::Axis::ROW);
                    temp_left_child.y_bootstrap = nc::append(temp_left_child.y_bootstrap, y_row, nc::Axis::ROW);
                }

                else {
                    temp_right_child.X_bootstrap = nc::append(temp_right_child.X_bootstrap, X_row, nc::Axis::ROW);
                    temp_right_child.y_bootstrap = nc::append(temp_right_child.y_bootstrap, y_row, nc::Axis::ROW);
                }

            }


            double split_info_gain = compute_information_gain(temp_left_child.y_bootstrap, temp_right_child.y_bootstrap);

            if (split_info_gain > best_info_gain) {
                // update details of best node to split at

                // save the children's bootstrapped data
                left_child->X_bootstrap = temp_left_child.X_bootstrap;
                left_child->y_bootstrap = temp_left_child.y_bootstrap;
                right_child->X_bootstrap = temp_right_child.X_bootstrap;
                right_child->y_bootstrap = temp_right_child.y_bootstrap;

                // save the children and other relevant info to the main node
                parent_node->information_gain = split_info_gain;
                parent_node->feature_idx = feat_idx;
                parent_node->split_point = split_point;

                best_info_gain = split_info_gain;
            }
        }
    }

    return;
}


void randomforest_classifier::split_node(rf_node* node, int max_features, int min_samples_split, int max_depth, int depth) {
    // splits a node. find_split() must be called on node before this function may be used.

    rf_node* left_child = node->get_leftchild(); // get pointers to the children
    rf_node* right_child = node->get_rightchild(); // get pointers to the children

    int left_n_samples = (left_child->y_bootstrap).reshape(-1, 1).shape().rows;
    int right_n_samples = (right_child->y_bootstrap).reshape(-1, 1).shape().rows;

    if (left_n_samples == 0 || right_n_samples == 0) {
        // if one of our children has no samples left, make the node a leaf node
        // delete the children as they're now redundant, and set node to be a leaf.
        delete node->get_leftchild();
        delete node->get_rightchild();
        
        node->leaf_value = calculate_leaf_value(node);
        node->is_leaf = true;
        
        return;
    }

    else if (depth >= max_depth) {
        // If we have hit the max depth, make the children leaves and return.

        left_child->leaf_value = calculate_leaf_value(left_child);
        left_child->is_leaf = true;

        right_child->leaf_value = calculate_leaf_value(right_child);
        right_child->is_leaf = true;
        return;
    }

    // std::cout << "LEFT_N_SAMPLES: " << left_n_samples << std::endl;
    // std::cout << "RIGHT_N_SAMPLES: " << right_n_samples << std::endl;

    // find the split points for the left and right children 
    find_split(left_child);
    find_split(right_child);

    if (left_child->X_bootstrap.shape().rows <= min_samples_split) {
        // if the left child has less samples than our min_samples_split, set the left child to be a leaf

        left_child->leaf_value = calculate_leaf_value(left_child);
        left_child->is_leaf = true;
    }

    else {
        // otherwise, split the left child further
        split_node(left_child, max_features, min_samples_split, max_depth, depth + 1);
    }

    if (right_child->X_bootstrap.shape().rows <= min_samples_split) {
        // if the right child has less samples than our min_samples_split, set the right child to be a leaf

        right_child->leaf_value = calculate_leaf_value(right_child);
        right_child->is_leaf = true;
    }

    else {
        // otherwise, split the right child further
        split_node(right_child, max_features, min_samples_split, max_depth, depth + 1);
    }
    
    return;

}


double randomforest_classifier::compute_information_gain(nc::NdArray<double>& lc_y_bootstrap, nc::NdArray<double>& rc_y_bootstrap) {
    /*
    Compute information gain

    Example
    ---------
    double split_info_gain = compute_information_gain(left_child.y_bootstrap, right_child.y_bootstrap);    
    */

    lc_y_bootstrap.reshape(-1, 1);
    rc_y_bootstrap.reshape(-1, 1);

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
        p_parent = (double)parent_class_0_count / (double)n_samples_parent;
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


double randomforest_classifier::compute_oob_score(rf_node* tree, nc::NdArray<double>& X_oob, nc::NdArray<double>& y_oob) {
    // compute the decision tree's score on the out-of-bag set
    int correct_labels = 0;
    int n_samples = X_oob.shape().rows;

    for (int i = 0; i < n_samples; i++) {
        double y_pred = predict_sample(tree, X_oob(i, X_oob.cSlice()));
        double y_true = y_oob(i, y_oob.cSlice())(0, 0);

        if (y_pred == y_true) {
            correct_labels += 1;
        }
    }

    return (double)correct_labels / (double)n_samples;
}


double randomforest_classifier::calculate_leaf_value(rf_node* node) {
    // this function is called when setting a node to be a leaf. Calculates
    // the most frequently occuring value in the node's data and makes that the leaf value.
    nc::NdArray<double> y_bootstrap = (*node).y_bootstrap;

    auto y_vec = y_bootstrap.toStlVector();

    double f = get_most_frequent_element(y_vec);

    return f;
}


double randomforest_classifier::predict_sample(rf_node* tree, nc::NdArray<double> X_sample) {
    /*
    use tree to predict label

    tree must already be an initialized node (i.e., split_node must have been called on this node already)

    Parameters
    ----------
    tree: rf_node that represents the root of the tree
    X_sample: nc::NdArray<double> of size (1, n_features) to predict on.

    Returns
    ---------
    leaf_value: the class predicted by the tree on sample X_sample
    */ 

    if (tree->is_leaf) {
        return tree->leaf_value;
    }

    else {

        X_sample.reshape(1, -1); // ensure sample is in row format

        int feature_idx = tree->feature_idx; // index of feature to decide on

        double val = X_sample(0, feature_idx);
        double sp = tree->split_point;

        // std::cout << val << " | " << sp << std::endl; 

        if (val <= sp) { // check if we want to go to the left or right child
            // if our value is <= the split point
            return predict_sample(tree->get_leftchild(), X_sample);
            }

        else {
            // if our value is > than the split point
            return predict_sample(tree->get_rightchild(), X_sample);
            }
    }
}
