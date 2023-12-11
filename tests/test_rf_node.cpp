// tests/test_metrics.cpp

#include <catch2/catch_test_macros.hpp>
#include "../include/rf_node.hpp"
#include "NumCpp.hpp"

TEST_CASE("rf_node test", "[RF NODE]") {

    rf_node node1(1);
    rf_node node2(2);
    rf_node node3(3);
    rf_node node4(4);

    node1.set_leftchild(&node2);
    node1.set_rightchild(&node3);

    node2.set_leftchild(&node4);

    REQUIRE( node1.get_data() == 1 );
    REQUIRE( node1.get_leftchild()->get_data() == 2 );
    REQUIRE( node1.get_rightchild()->get_data() == 3 );
    REQUIRE( node1.get_leftchild()->get_leftchild()->get_data() == 4 );

    nc::NdArray<double> a1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};


    nc::NdArray<int> a2 = {1, 2, 3, 1, 1, 1, 2, 3, 2};
    // nc::NdArray<double> preds;


    // preds = nc::append(preds, a2, nc::Axis::ROW);
    // preds = nc::append(preds, a2, nc::Axis::ROW);
    // preds = nc::append(preds, a2, nc::Axis::ROW);

}