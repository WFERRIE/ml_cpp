// tests/test_metrics.cpp

#include <catch2/catch_test_macros.hpp>
#include "../include/rf_node.hpp"
#include "NumCpp.hpp"

TEST_CASE("rf_node test", "[RF NODE]") {

    SECTION("Testing getters and setters on left and right children") {
        rf_node node1 = rf_node();
        rf_node node2 = rf_node();
        rf_node node3 = rf_node();
        rf_node node4 = rf_node();

        node1.data = 1;
        node2.data = 2;
        node3.data = 3;
        node4.data = 4;

        node1.set_leftchild(&node2);
        node1.set_rightchild(&node3);

        node2.set_leftchild(&node4);

        REQUIRE(node1.get_leftchild()->get_leftchild()->data == 4);
        REQUIRE(node1.get_leftchild()->data == 2);
        REQUIRE(node1.get_rightchild()->data == 3);

    }

    SECTION("Testing node destructor") {
        rf_node* node1 = new rf_node();
        rf_node* node2 = new rf_node();
        rf_node* node3 = new rf_node();
        rf_node* node4 = new rf_node();

        node1->data = 1;
        node2->data = 2;
        node3->data = 3;
        node4->data = 4;

        node1->set_leftchild(node2);
        node1->set_rightchild(node3);

        node2->set_leftchild(node4);

        delete node1;

    }

    SECTION("Testing overwriting bootstrap data of a node") {
        rf_node node5 = rf_node();
        rf_node temp_node5 = rf_node();

        nc::NdArray<double> a1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        nc::NdArray<double> a2 = {1, 2, 3, 1, 1, 1, 2, 3, 2};


        temp_node5.X_bootstrap = a1;
        node5.X_bootstrap = temp_node5.X_bootstrap;

        temp_node5.X_bootstrap = a2;

        int elements_correct = nc::sum<int>(nc::equal(node5.X_bootstrap, a1).astype<int>())(0, 0);

        REQUIRE(elements_correct == a1.shape().rows * a1.shape().cols);


    }
    

}