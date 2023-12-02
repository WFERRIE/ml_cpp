#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "NumCpp.hpp"
#include <string>

class linear_regression {

    private:
        nc::NdArray<double> weights;
        double bias;
        nc::uint32 n_classes;
        int max_iters;
        double lr;
        double tol;
        int init_mode;
        std::string penalty;
        double reg_strength;

        double compute_cost(nc::NdArray<double>& X, nc::NdArray<double>& y_pred);

        nc::NdArray<double> add_bias_feature(nc::NdArray<double>& X);

        nc::NdArray<double> calculate_gradient(nc::NdArray<double>& X, nc::NdArray<double>& y);


    public:

        linear_regression(const std::string penalty = "l2", const double reg_strength = 0.1, const int max_iters = 1000, const double lr = 0.01, const double tol = 0.0001, const int init_mode = 1);

        ~linear_regression();

        const nc::NdArray<double> get_weights() const;

        const double get_bias() const;

        void fit(nc::NdArray<double>& X, nc::NdArray<double>& y, bool verbose);

        nc::NdArray<double> predict(nc::NdArray<double>& X, bool bias_feature = false);  

};

#endif
