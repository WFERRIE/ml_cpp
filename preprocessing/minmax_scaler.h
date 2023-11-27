#ifndef PREPROCESSING_MINMAX_SCALER_H
#define PREPROCESSING_MINMAX_SCALER_H

#include "NumCpp.hpp"

class minmax_scaler {

    // private:


    public:

        

        minmax_scaler();

        ~minmax_scaler();

        void fit();

        void transform();

        void fit_transform();

        void inverse_transform();

    };

#endif
