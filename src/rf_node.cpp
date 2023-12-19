#include "../include/rf_node.hpp"
#include "NumCpp.hpp"
#include <iostream>
#include <fstream>
#include <sstream>


rf_node::rf_node() : leftchild(nullptr), rightchild(nullptr), parent(nullptr){
    // Constructor. By default, rf_node has no children or parent.
}


rf_node::~rf_node() {
    // destructor recursively deletes a node and its children
    if (leftchild != nullptr) {
        // if node has a leftchild, recursively call its destructor and then set the leftchild ptr to nullptr
        delete leftchild;
        leftchild = nullptr;
    }

    if (rightchild != nullptr) {
        // if node has a rightchild, recursively call its destructor and then set the leftchild ptr to nullptr
        delete rightchild;
        rightchild = nullptr;
    }

    rf_node* parent = this->get_parent(); // get a ptr to the parent so we can overwrite the parent's leftchild/rightchild to be nullptr

    if (this->child_assignment == 1) {
        // this node is the leftchild of its parent. Update parent's leftchild ptr to be nullptr
        parent->leftchild = nullptr;
    }

    else if (this->child_assignment == 2) {
        // this node is the rightchild of its parent. Update parent's rightchild ptr to be nullptr
        parent->rightchild = nullptr;
    }

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


// setters
void rf_node::set_parent(rf_node* parent) {
    // only to be called by the set_leftchild and set_rightchild functions.
    // otherwise, parent and child relationships can get messed up
    this->parent = parent;
}


void rf_node::set_leftchild(rf_node* leftchild) {
    // set leftchild the be the leftchild of this.
    // Set child_assignment to be 1 so we can identify leftchild as a leftchild during destruction.
    this->leftchild = leftchild;
    leftchild->child_assignment = 1;
    leftchild->set_parent(this);
}


void rf_node::set_rightchild(rf_node* rightchild) {
    // set rightchild to be the rightchild of this.
    // Set child_assignment to 2 so we can identify rightchild as a rightchild during destruction.
    this->rightchild = rightchild;
    rightchild->child_assignment = 2;
    rightchild->set_parent(this);
}




