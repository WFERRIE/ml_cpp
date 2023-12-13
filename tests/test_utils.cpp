// tests/test_utils.cpp

#include <catch2/catch_test_macros.hpp>
#include "../include/utils.hpp"
#include "../include/dataset.hpp"
#include <vector>
#include <map>

TEST_CASE("get_most_frequent_element test", "[UTILS]") {

    std::vector<int> vec = {1, 1, 2, 3, 4};

    int f = get_most_frequent_element(vec);

    REQUIRE ( f == 1 );

}

TEST_CASE("read_csv test", "[UTILS]") {
        
        auto data = read_csv("../data/iris_binary.csv", true);

        dataset DS = dataset(data);

        DS.train_test_split(0.8, false);

        // std::cout << DS.get_X() << std::endl;
        // std::cout << DS.get_y() << std::endl;

}