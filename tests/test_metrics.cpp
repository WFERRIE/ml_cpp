// tests/test_metrics.cpp

#include <catch2/catch_test_macros.hpp>
#include "../include/metrics.hpp"
#include "NumCpp.hpp"

TEST_CASE("accuracy_score Test", "[Metrics]") {
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


// TEST_CASE("confusion_matrix Test", "[Metrics]") {
//     // Test case 1
//     nc::NdArray<double> y_true1 = {0.0, 2.0, 1.0, 1.0, 0.0};
//     nc::NdArray<double> y_pred1 = {1.0, 0.0, 1.0, 0.0, 0.0};

//     nc::NdArray<double> c1 = { {1.0, 1.0, 0.0}, {1.0, 1.0, 0.0}, {1.0, 0.0, 0.0} };

//     auto c2 = confusion_matrix(y_true1.transpose(), y_pred1.transpose());

//     int elements_correct = nc::sum<int>(nc::equal(c1, c2).astype<int>())(0, 0);
    
//     REQUIRE( elements_correct == 9 );

//     // Test case 2
//     // nc::NdArray<double> y_true2 = {1.0, 1.0, 0.0, 1.0, 0.0};
//     // nc::NdArray<double> y_pred2 = {1.0, 1.0, 1.0, 0.0, 0.0};
//     // REQUIRE( accuracy_score(y_true2, y_pred2) == 0.6 );
//     // REQUIRE( accuracy_score(y_true2.transpose(), y_pred2.transpose()) == 0.6 );

//     // // Test case 3
//     // nc::NdArray<double> y_true3 = {1.0, 1.0, 1.0, 1.0, 1.0};
//     // nc::NdArray<double> y_pred3 = {0.0, 0.0, 0.0, 0.0, 0.0};
//     // REQUIRE( accuracy_score(y_true3, y_pred3) == 0.0 );
//     // REQUIRE( accuracy_score(y_true3.transpose(), y_pred3.transpose()) == 0.0 );

//     // // Test case 4
//     // nc::NdArray<double> y_true4 = {1.0, 1.0, 1.0, 1.0, 1.0};
//     // nc::NdArray<double> y_pred4 = {1.0, 1.0, 1.0, 1.0, 1.0};
//     // REQUIRE( accuracy_score(y_true4, y_pred4) == 1.0 );
//     // REQUIRE( accuracy_score(y_true4.transpose(), y_pred4.transpose()) == 1.0 );

// }





TEST_CASE("confusion_matrix test", "[Metrics]") {

    SECTION("Test case 1: Basic test") {
        nc::NdArray<double> y_true = {0.0, 2.0, 1.0, 1.0, 0.0};
        nc::NdArray<double> y_pred = {1.0, 0.0, 1.0, 0.0, 0.0};

        nc::NdArray<double> expected = {{1.0, 1.0, 0.0},
                                        {1.0, 1.0, 0.0},
                                        {1.0, 0.0, 0.0}};

        nc::NdArray<double> result = confusion_matrix(y_true, y_pred);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }

    SECTION("Test case 2: Perfect predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {0.0, 1.0, 2.0, 0.0, 1.0};

        nc::NdArray<double> expected = {{2.0, 0.0, 0.0},
                                        {0.0, 2.0, 0.0},
                                        {0.0, 0.0, 1.0}};

        nc::NdArray<double> result = confusion_matrix(y_true.transpose(), y_pred.transpose());

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);
    }

    SECTION("Test case 3: No correct predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {2.0, 0.0, 1.0, 2.0, 0.0};

        nc::NdArray<double> expected = {{0.0, 0.0, 2.0},
                                        {2.0, 0.0, 0.0},
                                        {0.0, 1.0, 0.0}};

        nc::NdArray<double> result = confusion_matrix(y_true.transpose(), y_pred.transpose());

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols); // No correct predictions
    }

    // SECTION("Test case 4: Class mismatch") {
    //     nc::NdArray<double> y_true = {0.0, 1.0, 3.0, 0.0, 1.0};
    //     nc::NdArray<double> y_pred = {2.0, 0.0, 1.0, 4.0, 0.0};

    //     nc::NdArray<double> expected = {{0.0, 0.0, 1.0},
    //                                     {1.0, 0.0, 1.0},
    //                                     {1.0, 1.0, 0.0}};

    //     nc::NdArray<double> result = confusion_matrix(y_true.transpose(), y_pred.transpose());

    //     int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

    //     REQUIRE(elements_correct == 0); // No correct predictions
    // }
}


// nc::NdArray<double> confusion_matrix(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

// nc::NdArray<double> f1_score(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

// nc::NdArray<double> precision_score(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

// nc::NdArray<double> recall_score(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

// // regression

// nc::NdArray<double> max_error(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

// nc::NdArray<double> mean_absolute_error(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);

// nc::NdArray<double> mean_squared_error(nc::NdArray<double> y_true, nc::NdArray<double> y_pred);