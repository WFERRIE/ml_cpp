#ifndef MODELS_KMEANS_H
#define MODELS_KMEANS_H

#include "NumCpp.hpp"

class kmeans {

    private:
        int n_clusters;
        int max_iter;
        double tol;
        nc::NdArray<nc::uint32> labels;
        nc::NdArray<double> centroids;

        nc::NdArray<double> initialize_centroids(nc::NdArray<double>& X, const int n_clusters);

        double calculate_distance(const nc::NdArray<double>& point1, const nc::NdArray<double>& point2);

        nc::NdArray<nc::uint32> assign_labels(nc::NdArray<double>& X,  nc::NdArray<double>& clusters);

        nc::NdArray<double> update_centroids(nc::NdArray<double>& X, const int n_clusters, nc::NdArray<nc::uint32> labels);

    
        

        
    public:

        kmeans(const int n_clusters, const int max_iter, const double tol);

        ~kmeans();

        nc::NdArray<nc::uint32> fit(nc::NdArray<double>& X, bool verbose);

        void predict(nc::NdArray<double>& X);  
        
};

#endif
