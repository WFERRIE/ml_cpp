// tests/test_metrics.cpp

#include <catch2/catch_test_macros.hpp>
#include "../include/metrics.hpp"
#include "NumCpp.hpp"

TEST_CASE("accuracy_score Test", "[Metrics]") {
    // Test case 1
    nc::NdArray<double> y_true1 = {1.0, 0.0, 1.0, 1.0, 0.0};
    nc::NdArray<double> y_pred1 = {1.0, 0.0, 1.0, 0.0, 0.0};
    REQUIRE( accuracy_score(y_true1, y_pred1) == 0.8 );
    REQUIRE( accuracy_score(y_true1, y_pred1) == 0.8 );

    // Test case 2
    nc::NdArray<double> y_true2 = {1.0, 1.0, 0.0, 1.0, 0.0};
    nc::NdArray<double> y_pred2 = {1.0, 1.0, 1.0, 0.0, 0.0};
    REQUIRE( accuracy_score(y_true2, y_pred2) == 0.6 );
    REQUIRE( accuracy_score(y_true2, y_pred2) == 0.6 );

    // Test case 3
    nc::NdArray<double> y_true3 = {1.0, 1.0, 1.0, 1.0, 1.0};
    nc::NdArray<double> y_pred3 = {0.0, 0.0, 0.0, 0.0, 0.0};
    REQUIRE( accuracy_score(y_true3, y_pred3) == 0.0 );
    REQUIRE( accuracy_score(y_true3, y_pred3) == 0.0 );

    // Test case 4
    nc::NdArray<double> y_true4 = {1.0, 1.0, 1.0, 1.0, 1.0};
    nc::NdArray<double> y_pred4 = {1.0, 1.0, 1.0, 1.0, 1.0};
    REQUIRE( accuracy_score(y_true4, y_pred4) == 1.0 );
    REQUIRE( accuracy_score(y_true4, y_pred4) == 1.0 );

}

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

        nc::NdArray<double> result = confusion_matrix(y_true, y_pred);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);
    }

    SECTION("Test case 3: No correct predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {2.0, 0.0, 1.0, 2.0, 0.0};

        nc::NdArray<double> expected = {{0.0, 0.0, 2.0},
                                        {2.0, 0.0, 0.0},
                                        {0.0, 1.0, 0.0}};

        nc::NdArray<double> result = confusion_matrix(y_true, y_pred);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols); // No correct predictions
    }

}



TEST_CASE("f1_score test", "[Metrics]") {

    SECTION("Test case 1: Basic test") {
        nc::NdArray<double> y_true = {0.0, 2.0, 1.0, 1.0, 0.0};
        nc::NdArray<double> y_pred = {1.0, 0.0, 1.0, 0.0, 0.0};

        nc::NdArray<double> expected = {0.4, 0.5, 0.0};

        nc::NdArray<double> result = f1_score(y_true, y_pred);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().cols);

    }

    SECTION("Test case 2: Perfect predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {0.0, 1.0, 2.0, 0.0, 1.0};

        nc::NdArray<double> expected = {1.0, 1.0, 1.0};

        nc::NdArray<double> result = f1_score(y_true, y_pred);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().cols);
    }

    SECTION("Test case 3: No correct predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {2.0, 0.0, 1.0, 2.0, 0.0};

        nc::NdArray<double> expected = {0.0, 0.0, 0.0};

        nc::NdArray<double> result = f1_score(y_true, y_pred);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().cols); // No correct predictions
    }

}


TEST_CASE("precision_score test", "[Metrics]") {

    SECTION("Test case 1: Basic test") {
        nc::NdArray<double> y_true = {0.0, 2.0, 1.0, 1.0, 0.0};
        nc::NdArray<double> y_pred = {1.0, 0.0, 1.0, 0.0, 0.0};

        nc::NdArray<double> expected = {1.0 / 3.0, 0.5, 0.0};

        nc::NdArray<double> result = precision_score(y_true, y_pred);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().cols);

    }

    SECTION("Test case 2: Perfect predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {0.0, 1.0, 2.0, 0.0, 1.0};

        nc::NdArray<double> expected = {1.0, 1.0, 1.0};

        nc::NdArray<double> result = precision_score(y_true, y_pred);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().cols);
    }

    SECTION("Test case 3: No correct predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {2.0, 0.0, 1.0, 2.0, 0.0};

        nc::NdArray<double> expected = {0.0, 0.0, 0.0};

        nc::NdArray<double> result = precision_score(y_true, y_pred);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().cols); // No correct predictions
    }

}


TEST_CASE("recall_score test", "[Metrics]") {

    SECTION("Test case 1: Basic test") {
        nc::NdArray<double> y_true = {0.0, 2.0, 1.0, 1.0, 0.0};
        nc::NdArray<double> y_pred = {1.0, 0.0, 1.0, 0.0, 0.0};

        nc::NdArray<double> expected = {0.5, 0.5, 0.0};

        nc::NdArray<double> result = recall_score(y_true, y_pred);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().cols);

    }

    SECTION("Test case 2: Perfect predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {0.0, 1.0, 2.0, 0.0, 1.0};

        nc::NdArray<double> expected = {1.0, 1.0, 1.0};

        nc::NdArray<double> result = recall_score(y_true, y_pred);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().cols);
    }

    SECTION("Test case 3: No correct predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {2.0, 0.0, 1.0, 2.0, 0.0};

        nc::NdArray<double> expected = {0.0, 0.0, 0.0};

        nc::NdArray<double> result = recall_score(y_true, y_pred);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().cols); // No correct predictions
    }

}

// // regression


TEST_CASE("max_error test", "[Metrics]") {

    SECTION("Test case 1: Basic test") {
        nc::NdArray<double> y_true = {0.0, 2.0, 1.0, 1.0, 0.0};
        nc::NdArray<double> y_pred = {1.0, 0.0, 1.0, 0.0, 0.0};

        double expected = 2.0;

        double result = max_error(y_true, y_pred);

        REQUIRE(expected == result);

    }

    SECTION("Test case 2: Perfect predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {0.0, 1.0, 2.0, 0.0, 1.0};

        double expected = 0.0;

        double result = max_error(y_true, y_pred);

        REQUIRE(expected == result);
    }

    SECTION("Test case 3: No correct predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {2.0, 0.0, 1.0, 2.0, 0.0};

        double expected = 1.0;

        double result = max_error(y_true, y_pred);

        REQUIRE(expected == result);
    }

}


TEST_CASE("mean_absolute_error test", "[Metrics]") {

    SECTION("Test case 1: Basic test") {
        nc::NdArray<double> y_true = {0.0, 2.0, 1.0, 1.0, 0.0};
        nc::NdArray<double> y_pred = {1.0, 0.0, 1.0, 0.0, 0.0};

        double expected = 0.8;

        double result = mean_absolute_error(y_true, y_pred);

        REQUIRE(expected == result);

    }

    SECTION("Test case 2: Perfect predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {0.0, 1.0, 2.0, 0.0, 1.0};

        double expected = 0.0;

        double result = mean_absolute_error(y_true, y_pred);

        REQUIRE(expected == result);
    }

    SECTION("Test case 3: No correct predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {2.0, 0.0, 1.0, 2.0, 0.0};

        double expected = 1.4;

        double result = mean_absolute_error(y_true, y_pred);

        REQUIRE(expected == result);
    }

}

TEST_CASE("mean_squared_error test", "[Metrics]") {

    SECTION("Test case 1: Basic test") {
        nc::NdArray<double> y_true = {0.0, 2.0, 1.0, 1.0, 0.0};
        nc::NdArray<double> y_pred = {1.0, 0.0, 1.0, 0.0, 0.0};

        double expected = 1.2;

        double result = mean_squared_error(y_true, y_pred);

        REQUIRE(expected == result);

    }

    SECTION("Test case 2: Perfect predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {0.0, 1.0, 2.0, 0.0, 1.0};

        double expected = 0.0;

        double result = mean_squared_error(y_true, y_pred);

        REQUIRE(expected == result);
    }

    SECTION("Test case 3: No correct predictions") {
        nc::NdArray<double> y_true = {0.0, 1.0, 2.0, 0.0, 1.0};
        nc::NdArray<double> y_pred = {2.0, 0.0, 1.0, 2.0, 0.0};

        double expected = 2.2;

        double result = mean_squared_error(y_true, y_pred);

        REQUIRE(expected == result);
    }

}

