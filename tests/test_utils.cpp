// tests/test_utils.cpp

#include <catch2/catch_test_macros.hpp>
#include "../include/utils.hpp"
#include <vector>
#include <map>

TEST_CASE("get_most_frequent_element test", "[UTILS]") {

    std::vector<int> vec = {1, 1, 2, 3, 4};

    int f = get_most_frequent_element(vec);

    REQUIRE ( f == 1 );

}