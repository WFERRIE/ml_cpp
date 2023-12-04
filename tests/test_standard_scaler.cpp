// tests/test_metrics.cpp

#include <catch2/catch_test_macros.hpp>
#include "../include/standard_scaler.hpp"
#include "NumCpp.hpp"


TEST_CASE("standard_scaler test", "[minmax_scaler]") {

    SECTION("Test case 1: fit_transform test") {
        nc::NdArray<double> X = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9}};

        nc::NdArray<double> expected = {{-1.22474, -1.22474, -1.22474},
                                        {0.0, 0.0, 0.0},
                                        {1.22474, 1.22474, 1.22474}};


        standard_scaler ss = standard_scaler();
        auto result = ss.fit_transform(X);

        int elements_correct = nc::sum<int>(nc::isclose(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }

    SECTION("Test case 2: fit_transform test 2") {
        nc::NdArray<double> X = {{0.0, -100, 5},
                                    {0.0, 0.0, 15},
                                    {0.0, 0.0, 15}};

        nc::NdArray<double> expected = {{0.0, -1.41421, -1.41421},
                                        {0.0, 0.70710, 0.70710},
                                        {0.0, 0.70710, 0.70710}};



        standard_scaler ss = standard_scaler();
        auto result = ss.fit_transform(X);

        int elements_correct = nc::sum<int>(nc::isclose(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }

    SECTION("Test case 3: fit_transform negative values ") {
        nc::NdArray<double> X = {{0.0, -100, -5},
                                    {0.0, 0.0, -15},
                                    {0.0, 0.0, -15}};

        nc::NdArray<double> expected = {{0.0, -1.41421, 1.41421},
                                        {0.0, 0.70710, -0.70710},
                                        {0.0, 0.70710, -0.70710}};


        standard_scaler ss = standard_scaler();
        auto result = ss.fit_transform(X);

        int elements_correct = nc::sum<int>(nc::isclose(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }

    SECTION("Test case 4: fit_transform negative values ") {
        nc::NdArray<double> X = {{-1, -100, -5},
                                    {-25, -73, -15},
                                    {-33, -1, -15}};

        nc::NdArray<double> expected = {{1.37281, -1.00514, 1.41421},
                                        {-0.392232, -0.358979, -0.707107},
                                        {-0.980581, 1.36412, -0.707107}};


        standard_scaler ss = standard_scaler();
        auto result = ss.fit_transform(X);

        int elements_correct = nc::sum<int>(nc::isclose(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }


    SECTION("Test case 5: fit and then transform test") {
        nc::NdArray<double> X = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9}};

        nc::NdArray<double> expected = {{-1.22474, -1.22474, -1.22474},
                                        {0.0, 0.0, 0.0},
                                        {1.22474, 1.22474, 1.22474}};


        standard_scaler ss = standard_scaler();
        ss.fit(X);
        auto result = ss.transform(X);

        int elements_correct = nc::sum<int>(nc::isclose(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }

    SECTION("Test case 6: fit and then transform test 2") {
        nc::NdArray<double> X = {{0.0, -100, 5},
                                    {0.0, 0.0, 15},
                                    {0.0, 0.0, 15}};

        nc::NdArray<double> expected = {{0.0, -1.41421, -1.41421},
                                        {0.0, 0.70710, 0.70710},
                                        {0.0, 0.70710, 0.70710}};


        standard_scaler ss = standard_scaler();
        ss.fit(X);
        auto result = ss.transform(X);

        int elements_correct = nc::sum<int>(nc::isclose(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }

    SECTION("Test case 7: fit and then transform negative values ") {
        nc::NdArray<double> X = {{0.0, -100, -5},
                                    {0.0, 0.0, -15},
                                    {0.0, 0.0, -15}};

        nc::NdArray<double> expected = {{0.0, -1.41421, 1.41421},
                                        {0.0, 0.70710, -0.70710},
                                        {0.0, 0.70710, -0.70710}};


        standard_scaler ss = standard_scaler();
        ss.fit(X);
        auto result = ss.transform(X);

        int elements_correct = nc::sum<int>(nc::isclose(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }

        SECTION("Test case 8: fit and then transform negative values ") {
        nc::NdArray<double> X = {{0.0, -100, -5},
                                    {0.0, 0.0, -15},
                                    {0.0, 0.0, -100}};

        nc::NdArray<double> expected = {{0.0, -1.41421, 0.821165},
                                        {0.0, 0.707107, 0.586546},
                                        {0.0, 0.707107, -1.40771}};


        standard_scaler ss = standard_scaler();
        ss.fit(X);
        auto result = ss.transform(X);

        int elements_correct = nc::sum<int>(nc::isclose(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }


}
