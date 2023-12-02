#ifndef MINMAX_SCALER_HPP
#define MINMAX_SCALER_HPP

#include "NumCpp.hpp"

class minmax_scaler {

    private:
        bool is_fit;
        const int feature_min;
        const int feature_max;
        nc::NdArray<double> min_vals;
        nc::NdArray<double> max_vals;
        

    public:

        minmax_scaler(int feature_min = 0, int feature_max = 1);

        ~minmax_scaler();

        void fit(nc::NdArray<double>& X);

        nc::NdArray<double> transform(nc::NdArray<double>& X);

        nc::NdArray<double> fit_transform(nc::NdArray<double>& X);

        nc::NdArray<double> inverse_transform(nc::NdArray<double>& X);
    };

#endif
