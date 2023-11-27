#ifndef PREPROCESSING_STANDARD_SCALER_H
#define PREPROCESSING_STANDARD_SCALER_H

#include "NumCpp.hpp"

class standard_scaler {

    private:
        nc::NdArray<double> means;
        nc::NdArray<double> stds;
        int n_features;
        int n_samples;
        bool is_fit;
        bool with_mean;
        bool with_std;
        

    public:

        standard_scaler(bool with_mean = true, bool with_std = true);

        ~standard_scaler();

        void fit(nc::NdArray<double>& X);

        nc::NdArray<double> transform(nc::NdArray<double>& X);

        nc::NdArray<double> fit_transform(nc::NdArray<double>& X);

        nc::NdArray<double> inverse_transform(nc::NdArray<double>& X);

        nc::NdArray<double> get_means();

        nc::NdArray<double> get_stds();
    };

#endif
