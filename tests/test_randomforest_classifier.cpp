#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../include/randomforest_classifier.hpp"
#include "NumCpp.hpp"
#include <iostream>
#include <tuple>
#include <set>
#include "../include/rf_node.hpp"


void my_function(rf_node* node) {
    rf_node* left_child = new rf_node();

    left_child->is_leaf = true;

    node->set_leftchild(left_child);
    node->information_gain = 5;

    return;

}

// should test what happens in bootstrap() if there if the oob set is of size 0
TEST_CASE("rfc Test", "[Randomforest Classifier]") {

    nc::NdArray<double> X = {{1, 0.5, 1.2}, {1, 0.8, 1.1}, {12, 14, 20}, {11, 13, 20}, {10, 13, 21}, {1.05, 1.03, 1.1},
                            {0.9, 0.33, 1.1}, {0.9, 0.7, 1.1}, {11, 16, 18}, {12, 18, 20}, {12, 17, 30}, {1, 1, 1.2},
                            {0.75, 0.3, 1.3}, {1.5, 1.5, 2}, {10, 15.1, 20}, {10.6, 11, 30}, {11, 18, 20}, {1.1, 1, 1.1},
                            {0.8, 0.25, 1.4}, {1.3, 0.9, 1.3}, {10.1, 17, 23}, {10.9, 15, 20}, {10, 15, 10}, {0.9, 0, 1.5}};
    nc::NdArray<double> y = {0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0};

    SECTION ("compute_entropy()") {
        
        randomforest_classifier rfc = randomforest_classifier(100, 10, 2, 2);
        
        REQUIRE( rfc.compute_entropy(0.0) == 0.0);
        REQUIRE( rfc.compute_entropy(1.0) == 0.0);
        REQUIRE( rfc.compute_entropy(0.5) == 1.0);
        REQUIRE_THAT( rfc.compute_entropy(0.1), Catch::Matchers::WithinRel(0.469, 0.001));


    }

    SECTION ("compute_information_gain") {

        nc::NdArray<double> y1 = {0, 0, 0, 0, 0, 0, 0, 1};

        nc::NdArray<double> y2 = {0, 0, 0, 0, 0, 0,
                                  1, 1, 1, 1, 1, 1};

        nc::NdArray<double> y3 = {0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1};
        nc::NdArray<double> y4 = {0, 0, 0};

        randomforest_classifier rfc = randomforest_classifier(100, 10, 2, 2);

        // REQUIRE_THAT( rfc.compute_information_gain(y1, y2), Catch::Matchers::WithinRel(0.117, 0.01));
        // REQUIRE_THAT( rfc.compute_information_gain(y3, y4), Catch::Matchers::WithinRel(0.079, 0.01));

    }


    SECTION (".fit()") {

        randomforest_classifier rfc = randomforest_classifier(100, 10, 2, 2);
        rfc.fit(X, y);

        // rf_node* root = new rf_node();

        // my_function(root);

        // std::cout << root->information_gain << std::endl;
        // std::cout << root->get_leftchild()->is_leaf << std::endl;

        

    }






    // auto [a1_bootstrapped, b1_bootstrapped, a1_oob, b1_oob] = rfc.bootstrap(a1, b1);

    // rfc.find_split(a1_bootstrapped, b1_bootstrapped, 2);
    
    
}

// int n_estimators;
// int max_features;
// int max_depth;
// int min_samples_split;
// std::vector<rf_node*> tree_list;
// std::vector<double> oob_list;

// std::tuple<nc::NdArray<double>, nc::NdArray<double>, nc::NdArray<double>, nc::NdArray<double>> bootstrap(nc::NdArray<double>& X, nc::NdArray<double>& y);

// double compute_information_gain(nc::NdArray<double>& lc_y_bootstrap, nc::NdArray<double>& rc_y_bootstrap);

// double compute_oob_score(rf_node* tree, nc::NdArray<double>& X_test, nc::NdArray<double>& y_test);

// rf_node find_split(nc::NdArray<double>& X_bootstrap, nc::NdArray<double>& y_bootstrap, int max_features);

// double calculate_terminal_node(rf_node* node);

// rf_node split_node(rf_node* node, int max_features, int min_samples_split, int max_depth, int depth);

// rf_node build_tree(nc::NdArray<double>& X_bootstrap, nc::NdArray<double>& y_bootstrap);

// double predict_tree(rf_node* tree, nc::NdArray<double> X_test);


// // public:

// randomforest_classifier(const int n_estimators = 100, const int max_depth = 10, const int min_samples_split = 2, int max_features = -1);

// ~randomforest_classifier();

// void fit(nc::NdArray<double>& X, nc::NdArray<double>& y, bool verbose);

// nc::NdArray<double> predict(nc::NdArray<double>& X); 