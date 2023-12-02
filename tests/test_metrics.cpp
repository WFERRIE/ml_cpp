// tests/test_metrics.cpp

#include <catch2/catch_test_macros.hpp>
#include "../include/metrics.hpp"
#include "NumCpp.hpp"

TEST_CASE("Accuracy Score Test", "[Metrics]") {
    // Test case 1
    nc::NdArray<double> y_true1 = {1.0, 0.0, 1.0, 1.0, 0.0};
    nc::NdArray<double> y_pred1 = {1.0, 0.0, 1.0, 0.0, 0.0};
    REQUIRE( accuracy_score(y_true1, y_pred1) == 0.8 );

    // Test case 2
    nc::NdArray<double> y_true2 = {1.0, 1.0, 0.0, 1.0, 0.0};
    nc::NdArray<double> y_pred2 = {1.0, 1.0, 1.0, 0.0, 0.0};
    REQUIRE( accuracy_score(y_true2, y_pred2) == 0.6 );

    // Add more test cases as needed
}
