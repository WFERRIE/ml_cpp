#ifndef RANDOMFOREST_CLASSIFIER_HPP
#define RANDOMFOREST_CLASSIFIER_HPP

#include "NumCpp.hpp"
#include <string>
#include <tuple>
#include "../include/rf_node.hpp"
class randomforest_classifier {

    public:

        int n_estimators;
        int max_features;
        int max_depth;
        int min_samples_split;

        double compute_entropy(double p);

        std::tuple<nc::NdArray<double>, nc::NdArray<double>, nc::NdArray<double>, nc::NdArray<double>> bootstrap(nc::NdArray<double>& X, nc::NdArray<double>& y);

        double compute_information_gain(nc::NdArray<double>& lc_y_bootstrap, nc::NdArray<double>& rc_y_bootstrap);

        void compute_oob_score();

        rf_node find_split(nc::NdArray<double>& X_bootstrap, nc::NdArray<double>& y_bootstrap, int max_features);

        void terminal_node();

        void split_node();
        
        void build_tree();

        void predict_tree();

        void predict_rf();



    // public:

        randomforest_classifier();

        ~randomforest_classifier();

        void fit(nc::NdArray<double>& X, nc::NdArray<double>& y, bool verbose);

        nc::NdArray<double> predict(nc::NdArray<double>& X);  

};

#endif
