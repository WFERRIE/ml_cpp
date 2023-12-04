// tests/test_metrics.cpp

#include <catch2/catch_test_macros.hpp>
#include "../include/minmax_scaler.hpp"
#include "NumCpp.hpp"


TEST_CASE("minmax_scaler test", "[minmax_scaler]") {

    SECTION("Test case 1: fit_transform test") {
        nc::NdArray<double> X = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9}};

        nc::NdArray<double> expected = {{0, 0, 0},
                                        {0.5, 0.5, 0.5},
                                        {1.0, 1.0, 1.0}};


        minmax_scaler mm = minmax_scaler();
        auto result = mm.fit_transform(X);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }

    SECTION("Test case 2: fit_transform test 2") {
        nc::NdArray<double> X = {{0, -100, 5},
                                    {0, 0, 15},
                                    {0, 0, 15}};

        nc::NdArray<double> expected = {{0, 0, 0},
                                        {0, 1, 1},
                                        {0, 1, 1}};


        minmax_scaler mm = minmax_scaler();
        auto result = mm.fit_transform(X);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }

    SECTION("Test case 3: fit_transform negative values ") {
        nc::NdArray<double> X = {{0, -100, -5},
                                    {0, 0, -15},
                                    {0, 0, -15}};

        nc::NdArray<double> expected = {{0, 0, 1},
                                        {0, 1, 0},
                                        {0, 1, 0}};


        minmax_scaler mm = minmax_scaler();
        auto result = mm.fit_transform(X);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }

    SECTION("Test case 4: fit_transform negative values ") {
        nc::NdArray<double> X = {{0, -100, -5},
                                    {0, 0, -15},
                                    {0, 0, -15}};

        nc::NdArray<double> expected = {{0, 0, 1},
                                        {0, 1, 0},
                                        {0, 1, 0}};


        minmax_scaler mm = minmax_scaler();
        auto result = mm.fit_transform(X);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }


    SECTION("Test case 5: fit and then transform test") {
        nc::NdArray<double> X = {{1, 2, 3},
                                 {4, 5, 6},
                                 {7, 8, 9}};

        nc::NdArray<double> expected = {{0, 0, 0},
                                        {0.5, 0.5, 0.5},
                                        {1.0, 1.0, 1.0}};


        minmax_scaler mm = minmax_scaler();
        mm.fit(X);
        auto result = mm.transform(X);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }

    SECTION("Test case 6: fit and then transform test 2") {
        nc::NdArray<double> X = {{0, -100, 5},
                                    {0, 0, 15},
                                    {0, 0, 15}};

        nc::NdArray<double> expected = {{0, 0, 0},
                                        {0, 1, 1},
                                        {0, 1, 1}};


        minmax_scaler mm = minmax_scaler();
        mm.fit(X);
        auto result = mm.transform(X);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }

    SECTION("Test case 7: fit and then transform negative values ") {
        nc::NdArray<double> X = {{0, -100, -5},
                                    {0, 0, -15},
                                    {0, 0, -15}};

        nc::NdArray<double> expected = {{0, 0, 1},
                                        {0, 1, 0},
                                        {0, 1, 0}};


        minmax_scaler mm = minmax_scaler();
        mm.fit(X);
        auto result = mm.transform(X);

        int elements_correct = nc::sum<int>(nc::equal(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }

        SECTION("Test case 8: fit and then transform negative values ") {
        nc::NdArray<double> X = {{0, -100, -5},
                                    {0, 0, -15},
                                    {0, 0, -100}};

        nc::NdArray<double> expected = {{0, 0, 1},
                                        {0, 1, 0.89473684},
                                        {0, 1, 0}};


        minmax_scaler mm = minmax_scaler();
        mm.fit(X);
        auto result = mm.transform(X);

        int elements_correct = nc::sum<int>(nc::isclose(expected, result).astype<int>())(0, 0);

        REQUIRE(elements_correct == expected.shape().rows * expected.shape().cols);

    }


}
