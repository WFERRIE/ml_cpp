#include <catch2/catch_test_macros.hpp>
#include "../include/randomforest_classifier.hpp"
#include "NumCpp.hpp"
#include <iostream>
#include <tuple>
#include <set>
#include "../include/rf_node.hpp"

// should test what happens in bootstrap() if there if the oob set is of size 0
TEST_CASE("rfc Test", "[Randomforest Classifier]") {
    // nc::NdArray<double> a1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}, {16, 17, 18}};
    // nc::NdArray<double> b1 = {-1, -2, -3, -4, -5, -6};

    // randomforest_classifier rfc = randomforest_classifier();

    nc::NdArray<double> y = {{1, 0, 0, 1, 0, 0}};

    std::cout << nc::sum<int>(nc::where(y == 0.0, 1, 0)) << std::endl; 



    // auto [a1_bootstrapped, b1_bootstrapped, a1_oob, b1_oob] = rfc.bootstrap(a1, b1);

    // rfc.find_split(a1_bootstrapped, b1_bootstrapped, 2);
    
    
}
