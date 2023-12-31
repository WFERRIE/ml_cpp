#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include "NumCpp.hpp"
#include <string>

class logistic_regression {

    private:
        nc::NdArray<double> weights;
        nc::NdArray<double> bias;
        nc::uint32 n_classes;
        int max_iters;
        double lr;
        double tol;
        int init_mode;
        std::string penalty;
        double reg_strength;

        bool is_fit = false;

        double compute_BCE_cost(nc::NdArray<double>& predictions, nc::NdArray<double>& y);

        nc::NdArray<double> sigmoid(nc::NdArray<double>& z);


    public:

        logistic_regression(const std::string penalty = "l2", const double reg_strength = 0.1, const int max_iters = 1000, const double lr = 0.01, const double tol = 0.0001, const int init_mode = 1);

        ~logistic_regression();

        const nc::NdArray<double> get_weights() const;

        const nc::NdArray<double> get_bias() const;

        void fit(nc::NdArray<double>& X, nc::NdArray<double>& y, bool verbose);

        nc::NdArray<double> predict(nc::NdArray<double>& X);  

};

#endif
