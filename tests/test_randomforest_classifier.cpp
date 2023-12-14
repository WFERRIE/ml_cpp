#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../include/randomforest_classifier.hpp"
#include "NumCpp.hpp"
#include <iostream>
#include <tuple>
#include <set>
#include "../include/rf_node.hpp"
#include "../include/utils.hpp"
#include "../include/dataset.hpp"
#include "../include/metrics.hpp"


void my_function(rf_node* node) {
    rf_node* left_child = new rf_node();

    left_child->is_leaf = true;

    node->set_leftchild(left_child);
    node->information_gain = 5;

    return;

}

// should test what happens in bootstrap() if there if the oob set is of size 0
TEST_CASE("rfc Test", "[Randomforest Classifier]") {

    // nc::NdArray<double> X = {{1, 0.5, 1.2}, {1, 0.8, 1.1}, {12, 14, 20}, {11, 13, 20}, {10, 13, 21}, {1.05, 1.03, 1.1},
    //                         {0.9, 0.33, 1.1}, {0.9, 0.7, 1.1}, {11, 16, 18}, {12, 18, 20}, {12, 17, 30}, {1, 1, 1.2},
    //                         {0.75, 0.3, 1.3}, {1.5, 1.5, 2}, {10, 15.1, 20}, {10.6, 11, 30}, {11, 18, 20}, {1.1, 1, 1.1},
    //                         {0.8, 0.25, 1.4}, {1.3, 0.9, 1.3}, {10.1, 17, 23}, {10.9, 15, 20}, {10, 15, 10}, {0.9, 0, 1.5}};
    // nc::NdArray<double> y = {0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0};

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

        nc::NdArray<double> y5 = {1, 1, 1};
        nc::NdArray<double> y6 = {};

        randomforest_classifier rfc = randomforest_classifier(100, 10, 2, 2);

        REQUIRE_THAT( rfc.compute_information_gain(y1, y2), Catch::Matchers::WithinRel(0.117, 0.01));
        REQUIRE_THAT( rfc.compute_information_gain(y2, y1), Catch::Matchers::WithinRel(0.117, 0.01));
        REQUIRE_THAT( rfc.compute_information_gain(y3, y4), Catch::Matchers::WithinRel(0.079, 0.01));
        REQUIRE_THAT( rfc.compute_information_gain(y4, y3), Catch::Matchers::WithinRel(0.079, 0.01));

        REQUIRE( rfc.compute_information_gain(y5, y6) == 0.0 );
        REQUIRE( rfc.compute_information_gain(y6, y5) == 0.0 );



    }

    SECTION ("find_split()") {
        int n_estimators = 100;
        int max_depth = 10;
        int min_samples_split = 5;

        randomforest_classifier rfc = randomforest_classifier(n_estimators, max_depth, min_samples_split);
        
        rfc.max_features = 2;
        
        rf_node node = rf_node();

        nc::NdArray<double> X_b = {{4.3, 3, 1.1, 0.1}, {4.3, 3, 1.1, 0.1}, {4.3, 3, 1.1, 0.1}};
        nc::NdArray<double> y_b = {1, 1, 1};
        y_b.reshape(-1, 1);

        node.X_bootstrap = X_b;
        node.y_bootstrap = y_b;

        rfc.find_split(&node);

        REQUIRE ( node.feature_idx == 1 );

    }


    SECTION (".fit()") {

        nc::NdArray<double> data = read_csv("../data/tree_data_fixed.csv", false);

        dataset DS = dataset(data);

        DS.train_test_split(0.2, true);

        nc::NdArray<double> X_train = DS.get_X_train();
        nc::NdArray<double> y_train = DS.get_y_train();

        nc::NdArray<double> X_test = DS.get_X_test();
        nc::NdArray<double> y_test = DS.get_y_test();



        int n_estimators = 4;
        int max_depth = 5;
        int min_samples_split = 10;
        int max_features = 5;



        randomforest_classifier rfc = randomforest_classifier(n_estimators, max_depth, min_samples_split, max_features);
        


        // rfc.test_func();
        rfc.fit(X_train, y_train);


        for (int i = 0; i < n_estimators; i++) {
            auto n = rfc.tree_list[i];
            auto oob = rfc.oob_list[i];
            std::cout << "root.feature_idx: " << n->feature_idx << std::endl;
            std::cout << "root.information_gain: " << n->information_gain << std::endl;
            std::cout << "root.split_point: " << n->split_point << std::endl;
            std::cout << "oob score: " << oob << std::endl;
            std::cout << "\n" << std::endl;

        }

        auto y_pred = rfc.predict(X_test);

        std::cout << "f1_score: " << f1_score(y_test, y_pred) << std::endl;
        std::cout << "accuracy_score: " << accuracy_score(y_test, y_pred) << std::endl;

        

    }

    
}
