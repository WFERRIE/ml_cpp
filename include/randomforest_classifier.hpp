#ifndef RANDOMFOREST_CLASSIFIER_HPP
#define RANDOMFOREST_CLASSIFIER_HPP

#include "NumCpp.hpp"
#include <string>
#include <tuple>
#include <vector>
#include "../include/rf_node.hpp"
class randomforest_classifier {

    public:
        randomforest_classifier(const int n_estimators = 10, const int max_depth = 5, const int min_samples_split = 5, int max_features = -1);
        ~randomforest_classifier();
        void fit(nc::NdArray<double>& X, nc::NdArray<double>& y);
        nc::NdArray<double> predict(nc::NdArray<double>& X);


    private:
        int n_estimators;
        int max_features;
        int max_depth;
        int min_samples_split;
        std::vector<rf_node*> tree_list;
        std::vector<double> oob_list;
        bool is_fit = false;

        double compute_entropy(double p);
        std::tuple<nc::NdArray<double>, nc::NdArray<double>, nc::NdArray<double>, nc::NdArray<double>> bootstrap(nc::NdArray<double>& X, nc::NdArray<double>& y);
        double compute_information_gain(nc::NdArray<double>& lc_y_bootstrap, nc::NdArray<double>& rc_y_bootstrap);
        double compute_oob_score(rf_node* tree, nc::NdArray<double>& X_oob, nc::NdArray<double>& y_oob);
        void find_split(rf_node* node);
        double calculate_leaf_value(rf_node* node);
        void split_node(rf_node* node, int max_features, int min_samples_split, int max_depth, int depth);
        void build_tree(rf_node* root, nc::NdArray<double>& X_bootstrap, nc::NdArray<double>& y_bootstrap);
        double predict_sample(rf_node* tree, nc::NdArray<double> X_test);  
};

#endif
