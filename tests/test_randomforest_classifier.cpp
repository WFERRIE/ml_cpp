#include <catch2/catch_test_macros.hpp>
#include "../include/randomforest_classifier.hpp"
#include "NumCpp.hpp"
#include <iostream>
#include <tuple>
#include <set>

// should test what happens in bootstrap() if there if the oob set is of size 0
TEST_CASE("rfc Test", "[Randomforest Classifier]") {
    nc::NdArray<double> a1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}, {16, 17, 18}};
    nc::NdArray<double> b1 = {-1, -2, -3, -4, -5, -6};
    // b1.reshape(-1, 1);

    // auto [a1_bootstrapped, b1_bootstrapped] = bootstrap(a1, b1);
    // auto [a2_bootstrapped, b2_bootstrapped] = bootstrap(a1, b1);
    // auto [a3_bootstrapped, b3_bootstrapped] = bootstrap(a1, b1);

    randomforest_classifier rfc = randomforest_classifier();


    auto [a1_bootstrapped, b1_bootstrapped, a1_oob, b1_oob] = rfc.bootstrap(a1, b1);

    std::cout << a1_bootstrapped << std::endl;
    std::cout << b1_bootstrapped << std::endl;
    std::cout << a1_oob << std::endl;
    std::cout << b1_oob << std::endl;
    
}


    // int n_samples = input.shape().rows;

    // for (int i = n_samples - 1; i > 0; i--) {
    //     int j = std::rand() % (i + 1);

    //     nc::NdArray<double> temp = input(j, input.cSlice());
    //     input.put(j, input.cSlice(), input(i, input.cSlice())); 
    //     input.put(i, input.cSlice(), temp); 

    // }

    // return input;