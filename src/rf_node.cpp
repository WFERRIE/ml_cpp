#include "../include/rf_node.hpp"
#include "NumCpp.hpp"
#include <iostream>
#include <fstream>
#include <sstream>



rf_node::rf_node(int data) : data(data), parent(nullptr), leftchild(nullptr), rightchild(nullptr) {
    // Constructor implementation...
}
rf_node::~rf_node() {

}

// getters
rf_node* rf_node::get_parent() const {
    return parent;
}

rf_node* rf_node::get_leftchild() const {
    return leftchild;
}

rf_node* rf_node::get_rightchild() const {
    return rightchild;
}

int rf_node::get_data() const {
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