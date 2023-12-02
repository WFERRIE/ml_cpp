// tests/test_metrics.cpp

#include <catch2/catch_test_macros.hpp>
#include "../include/metrics.hpp"
#include "NumCpp.hpp"

TEST_CASE("Accuracy Score Test", "[Metrics]") {
    // Test case 1
    nc::NdArray<double> y_true1 = {1.0, 0.0, 1.0, 1.0, 0.0};
    nc::NdArray<double> y_pred1 = {1.0, 0.0, 1.0, 0.0, 0.0};
    REQUIRE( accuracy_score(y_true1, y_pred1) == 0.8 );
    REQUIRE( accuracy_score(y_true1.transpose(), y_pred1.transpose()) == 0.8 );

    // Test case 2
    nc::NdArray<double> y_true2 = {1.0, 1.0, 0.0, 1.0, 0.0};
    nc::NdArray<double> y_pred2 = {1.0, 1.0, 1.0, 0.0, 0.0};
    REQUIRE( accuracy_score(y_true2, y_pred2) == 0.6 );
    REQUIRE( accuracy_score(y_true2.transpose(), y_pred2.transpose()) == 0.6 );

    // Test case 3
    nc::NdArray<double> y_true3 = {1.0, 1.0, 1.0, 1.0, 1.0};
    nc::NdArray<double> y_pred3 = {0.0, 0.0, 0.0, 0.0, 0.0};
    REQUIRE( accuracy_score(y_true3, y_pred3) == 0.0 );
    REQUIRE( accuracy_score(y_true3.transpose(), y_pred3.transpose()) == 0.0 );

    // Test case 4
    nc::NdArray<double> y_true4 = {1.0, 1.0, 1.0, 1.0, 1.0};
    nc::NdArray<double> y_pred4 = {1.0, 1.0, 1.0, 1.0, 1.0};
    REQUIRE( accuracy_score(y_true4, y_pred4) == 1.0 );
    REQUIRE( accuracy_score(y_true4.transpose(), y_pred4.transpose()) == 1.0 );



}
