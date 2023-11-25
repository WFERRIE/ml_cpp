#ifndef MODELS_KMEANS_H
#define MODELS_KMEANS_H

#include "NumCpp.hpp"
#include <vector>

class kmeans {

    private:
        int n_clusters;
        int max_iter;
        double tol;
        std::vector<nc::NdArray<double>> centroids;
        nc::NdArray<double> labels;

        std::vector<nc::NdArray<double>> initialize_centroids(nc::NdArray<double>& X, const int n_clusters);

        double calculate_distance(const nc::NdArray<double>& point1, const nc::NdArray<double>& point2);

        void assign_labels();

        void update_centroids();

        
    public:

        kmeans(const int n_clusters, const int max_iter, const double tol);

        ~kmeans();

        void fit(nc::NdArray<double>& X, bool verbose);

        void predict(nc::NdArray<double>& X);  

};

#endif
