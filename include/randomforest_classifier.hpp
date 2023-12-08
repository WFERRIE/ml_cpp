#ifndef RANDOMFOREST_CLASSIFIER_HPP
#define RANDOMFOREST_CLASSIFIER_HPP

#include "NumCpp.hpp"
#include <string>
#include <tuple>

class randomforest_classifier {

    private:

        int n_estimators;
        int max_features;
        int max_depth;
        int min_samples_split;

        double compute_entropy(double p);

        void compute_information_gain();

        void compute_oob_score();

        void find_split();

        void terminal_node();

        void split_node();
        
        void build_tree();

        void predict_tree();

        void predict_rf();



    public:

        randomforest_classifier();

        ~randomforest_classifier();

        std::tuple<nc::NdArray<double>, nc::NdArray<double>, nc::NdArray<double>, nc::NdArray<double>> bootstrap(nc::NdArray<double>& X, nc::NdArray<double>& y);
        void fit(nc::NdArray<double>& X, nc::NdArray<double>& y, bool verbose);

        nc::NdArray<double> predict(nc::NdArray<double>& X);  

};

#endif
