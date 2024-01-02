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

TEST_CASE("rfc Test", "[Randomforest Classifier]") {
    SECTION (".fit()") {

        // test model is able to learn some data to a high accuracy

        nc::NdArray<double> data = read_csv("../data/tree_data_fixed.csv", false);

        dataset DS = dataset(data);

        DS.train_test_split(0.05, true);

        nc::NdArray<double> X_train = DS.get_X_train();
        nc::NdArray<double> y_train = DS.get_y_train();

        nc::NdArray<double> X_test = DS.get_X_test();
        nc::NdArray<double> y_test = DS.get_y_test();



        int n_estimators = 4;
        int max_depth = 5;
        int min_samples_split = 10;
        int max_features = 5;
        bool verbose = false;

        randomforest_classifier rfc = randomforest_classifier(n_estimators, max_depth, min_samples_split, max_features);
    
        rfc.fit(X_train, y_train, verbose);

        auto y_pred = rfc.predict(X_test);

        double acc = accuracy_score(y_test, y_pred);

        REQUIRE(acc >= 0.99);

    }
}
