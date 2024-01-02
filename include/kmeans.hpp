#ifndef KMEANS_HPP
#define KMEANS_HPP

#include "NumCpp.hpp"

class kmeans {

    private:
        int n_clusters;
        int max_iter;
        double tol;
        nc::NdArray<double> centroids;
        nc::NdArray<double> prev_centroids;
        bool is_fit = false;


        nc::NdArray<double> initialize_centroids(nc::NdArray<double>& X, const int n_clusters);

        double calculate_distance(const nc::NdArray<double>& point1, const nc::NdArray<double>& point2);

        nc::NdArray<int> assign_labels(nc::NdArray<double>& X, nc::NdArray<double>& clusters);

        nc::NdArray<double> update_centroids(nc::NdArray<double>& X, nc::NdArray<int>& labels);

        
    public:

        kmeans(const int n_clusters, const int max_iter, const double tol);

        ~kmeans();

        void fit(nc::NdArray<double>& X, bool verbose);

        nc::NdArray<int> predict(nc::NdArray<double>& X);  

        nc::NdArray<double> get_centroids();
        
};

#endif
