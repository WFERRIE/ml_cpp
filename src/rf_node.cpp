#include "../include/rf_node.hpp"
#include "NumCpp.hpp"
#include <iostream>
#include <fstream>
#include <sstream>



rf_node::rf_node() : leftchild(nullptr), rightchild(nullptr) {
    // Constructor
}
rf_node::~rf_node() {
    // std::cout << "NODE IS BEING DESTROYED" << std::endl;
}

// getters
rf_node* rf_node::get_parent() {
    return parent;
}

rf_node* rf_node::get_leftchild() {
    return leftchild;
}

rf_node* rf_node::get_rightchild() {
    return rightchild;
}

int rf_node::get_data() {
    return data;
}







// setters
void rf_node::set_parent(rf_node* parent) {
    this->parent = parent;
}

void rf_node::set_leftchild(rf_node* leftchild) {
    this->leftchild = leftchild;
}

void rf_node::set_rightchild(rf_node* rightchild) {
    this->rightchild = rightchild;
}




