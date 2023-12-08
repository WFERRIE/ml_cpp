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
        int split_point;
        int feature_idx;

    public:
        rf_node(int data);
        ~rf_node();
        
        // getters
        rf_node* get_parent() const;
        rf_node* get_leftchild() const;
        rf_node* get_rightchild() const;
        int get_data() const;

        
        // setters
        void set_parent(rf_node* parent);
        void set_leftchild(rf_node* leftchild);
        void set_rightchild(rf_node* rightchild);

};

#endif