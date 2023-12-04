// tests/test_metrics.cpp

#include <catch2/catch_test_macros.hpp>
#include "../include/dataset.hpp"
#include "NumCpp.hpp"

TEST_CASE("dataset test", "[Dataset]") {

    SECTION("Test case 1: Basic test") {

        nc::NdArray<double> some_data = {{1, 1, 1, 1, -1},
                                {2, 2, 2, 2, -2},
                                {3, 3, 3, 3, -3},
                                {4, 4, 4, 4, -4},
                                {5, 5, 5, 5, -5},
                                {6, 6, 6, 6, -6},
                                {7, 7, 7, 7, -7},
                                {8, 8, 8, 8, -8}};

        dataset DS = dataset(some_data);

        DS.train_test_split(0.8, false);

        nc::NdArray<double> expected_X_train = {{1, 1, 1, 1},
                                                {2, 2, 2, 2},
                                                {3, 3, 3, 3},
                                                {4, 4, 4, 4},
                                                {5, 5, 5, 5},
                                                {6, 6, 6, 6}};

        nc::NdArray<double> expected_y_train = {{-1, -2, -3, -4, -5, -6}};

        nc::NdArray<double> expected_X_test = {{7, 7, 7, 7},
                                                {8, 8, 8, 8}};

        nc::NdArray<double> expected_y_test = {{-7, -8}};



        nc::NdArray<double> X_train = DS.get_X_train();
        nc::NdArray<double> y_train = DS.get_y_train();
        nc::NdArray<double> X_test = DS.get_X_test();
        nc::NdArray<double> y_test = DS.get_y_test();


        int elements_correct_X_train = nc::sum<int>(nc::equal(expected_X_train, X_train).astype<int>())(0, 0);
        int elements_correct_y_train = nc::sum<int>(nc::equal(expected_y_train.transpose(), y_train).astype<int>())(0, 0);
        int elements_correct_X_test = nc::sum<int>(nc::equal(expected_X_test, X_test).astype<int>())(0, 0);
        int elements_correct_y_test = nc::sum<int>(nc::equal(expected_y_test.transpose(), y_test).astype<int>())(0, 0);

        REQUIRE(elements_correct_X_train == expected_X_train.shape().rows * expected_X_train.shape().cols);
        REQUIRE(elements_correct_y_train == expected_y_train.shape().rows * expected_y_train.shape().cols);
        REQUIRE(elements_correct_X_test == expected_X_test.shape().rows * expected_X_test.shape().cols);
        REQUIRE(elements_correct_y_test == expected_y_test.shape().rows * expected_y_test.shape().cols);

    }

}


// class dataset {

//     public:
//         // constructor
//         dataset(nc::NdArray<double> data); // sets X and y
//         ~dataset();

//         void train_test_split(double train_size, bool shuffle);
//         // shuffles and splits X and y into X_train and y_train and X_test, and y_test

//         void set_X(nc::NdArray<double>& new_X);
//         void set_y(nc::NdArray<double>& new_y);

//         nc::NdArray<double> get_X();
//         nc::NdArray<double> get_y();

    
//     private:
//         nc::NdArray<double> X_train;
//         nc::NdArray<double> X_test;
//         nc::NdArray<double> y_train;
//         nc::NdArray<double> y_test;
//         nc::NdArray<double> X;
//         nc::NdArray<double> y;
//         nc::NdArray<double> data;

//         bool train_test_set = false;

//         nc::NdArray<double> fisher_yates_shuffle(nc::NdArray<double>& input);


// };