#ifndef MINMAX_SCALER_HPP
#define MINMAX_SCALER_HPP

#include "NumCpp.hpp"

class minmax_scaler {

    private:
        bool is_fit = false;
        const double feature_min;
        const double feature_max;
        nc::NdArray<double> min_vals;
        nc::NdArray<double> max_vals;
        

    public:

        minmax_scaler(double feature_min = 0.0, double feature_max = 1.0);

        ~minmax_scaler();

        void fit(nc::NdArray<double>& X);

        nc::NdArray<double> transform(nc::NdArray<double>& X);

        nc::NdArray<double> fit_transform(nc::NdArray<double>& X);

        nc::NdArray<double> inverse_transform(nc::NdArray<double>& X);

        nc::NdArray<double> get_min_vals();

        nc::NdArray<double> get_max_vals();
    };

#endif
