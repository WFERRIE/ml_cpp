#ifndef MODELS_LOGISTIC_REGRESSION_H
#define MODELS_LOGISTIC_REGRESSION_H

#include "NumCpp.hpp"

class logistic_regression {

    private:
        nc::NdArray<double> weights;
        nc::NdArray<double> bias;
        nc::uint32 n_classes;
        int n_iters;
        double lr;
        double tol = 1e-06;
        int init_mode;

        double compute_BCE_cost(nc::NdArray<double> predictions, nc::NdArray<double> y);

        nc::NdArray<double> sigmoid(nc::NdArray<double> z);


    public:

        logistic_regression(const int& n_iters, const double& lr, const int& init_mode);

        ~logistic_regression();

        const nc::NdArray<double>& get_weights() const;

        const nc::NdArray<double>& get_bias() const;

        void fit(nc::NdArray<double> X, nc::NdArray<double> y, bool verbose);

        nc::NdArray<double> predict(nc::NdArray<double> X);  

};

#endif
