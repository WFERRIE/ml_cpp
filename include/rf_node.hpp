#ifndef RF_NODE_HPP
#define RF_NODE_HPP

#include <vector>
#include <string>
#include "NumCpp.hpp"

class rf_node {
    private:
        rf_node* parent;
        rf_node* leftchild;
        rf_node* rightchild;

        int child_assignment = 0;
        // the child_assignment value is used in the destructor. A value of 0 indicates that
        // this node is not the child of any other node. A value of 1 indicates it is the left child
        // of another node. A 2 indicates it is the right child of another node.

        void set_parent(rf_node* parent);


    public:
        int data = -1; // unused attribute - useful for debugging and testing.
        rf_node();
        ~rf_node();
        
        // getters
        rf_node* get_parent();
        rf_node* get_leftchild();
        rf_node* get_rightchild();


        // probably is best to keep X and y bootstrap private and access via getters and setters
        // but im only using this rf_node class within the private methods of another class,
        // which means these are kind of private anyways, so im going to just make them public 
        // here to make the randomforest code cleaner
        nc::NdArray<double> X_bootstrap;
        nc::NdArray<double> y_bootstrap;

        double split_point;
        int feature_idx = -1;
        double information_gain;
        bool is_leaf = false;
        double leaf_value = -1;

        
        // setters
        void set_leftchild(rf_node* leftchild);
        void set_rightchild(rf_node* rightchild);



};

#endif