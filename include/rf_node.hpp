#ifndef RF_NODE_HPP
#define RF_NODE_HPP

#include <vector>
#include <string>
#include "NumCpp.hpp"

class rf_node {
    private:
        int data;
        rf_node* parent;
        rf_node* leftchild;
        rf_node* rightchild;

        
        

    public:

        rf_node(int data);
        ~rf_node();
        
        // getters
        rf_node* get_parent();
        rf_node* get_leftchild();
        rf_node* get_rightchild();
        int get_data();


        // probably is best to keep X and y bootstrap private and access via getters and setters
        // but im only using this rf_node class within the private methods of another class,
        // which means these are kind of private anyways, so im going to just make them public 
        // here to make the randomforest code cleaner
        nc::NdArray<double> X_bootstrap;
        nc::NdArray<double> y_bootstrap;
        int split_point;
        int feature_idx;
        double information_gain;
        bool is_leaf = false;
        double leaf_value = -1;

        
        // setters
        void set_parent(rf_node* parent);
        void set_leftchild(rf_node* leftchild);
        void set_rightchild(rf_node* rightchild);



};

#endif