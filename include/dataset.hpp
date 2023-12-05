#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>
#include "NumCpp.hpp"

class dataset {

    private:
        nc::NdArray<double> X_train;
        nc::NdArray<double> X_test;
        nc::NdArray<double> y_train;
        nc::NdArray<double> y_test;
        nc::NdArray<double> X;
        nc::NdArray<double> y;
        nc::NdArray<double> data;

        bool train_test_set = false;

        nc::NdArray<double> fisher_yates_shuffle(nc::NdArray<double>& input);

    public:
        // constructor
        dataset(nc::NdArray<double> data); // sets X and y
        ~dataset();

        void train_test_split(double train_size, bool shuffle);

        void set_X(nc::NdArray<double>& new_X);
        void set_y(nc::NdArray<double>& new_y);

        nc::NdArray<double> get_X();
        nc::NdArray<double> get_y();

        nc::NdArray<double> get_X_train();
        nc::NdArray<double> get_y_train();

        nc::NdArray<double> get_X_test();
        nc::NdArray<double> get_y_test();


};

#endif