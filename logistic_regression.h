#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "NumCpp.hpp"

class logistic_regression {

    private:
        nc::NdArray<double> weights;
        double bias = 0.0;
        int n_iters;
        double lr;

        double compute_BCE_cost(nc::NdArray<double> predictions, nc::NdArray<double> y);

        nc::NdArray<double> sigmoid(nc::NdArray<double> z);


    public:

        logistic_regression(const int& n_iters, const double& lr);

        ~logistic_regression();

        const nc::NdArray<double>& get_weights() const;

        const double& get_bias() const;


        void fit(nc::NdArray<double> X, nc::NdArray<double> y, bool verbose);

        nc::NdArray<double> predict(nc::NdArray<double> X);  

};

#endif
