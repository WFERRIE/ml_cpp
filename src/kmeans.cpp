#include "NumCpp.hpp"
#include <iostream>
#include "../include/kmeans.hpp"


kmeans::kmeans(const int n_clusters, const int max_iter, const double tol) : n_clusters(n_clusters), max_iter(max_iter), tol(tol){
    // constructor
}

kmeans::~kmeans() {
    // destructor
}


nc::NdArray<double> kmeans::initialize_centroids(nc::NdArray<double>& X, const int n_clusters) {
    /*
    Initializes n_clusters centroids. This is done by picking n_clusters random samples from X and assigning those
    as the initial centroid locations.

    Parameters 
    ---------
    X: The data on which to perform kmeans clustering.
    n_clusters: The number of clusters to produce (i.e. K).


    Returns 
    ----------
    centroids: an array of size (n_clusters, X.shape().cols) containing the 
    location of the initialized centroids.
    */

    int n_samples = X.shape().rows;
    int n_features = X.shape().cols;

    if (n_samples < n_clusters) {
        std::runtime_error("Error: More clusters were requested than there are data points. n_clusters must be less than X.shape().rows.");
    }

    nc::NdArray<double> centroids = nc::zeros<double>(n_clusters, n_features);

    for (int i = 0; i < n_clusters; i++) {
        int randInt = nc::random::randInt<int>(0, n_samples); // pick a integer between 0 and the number of data points in X
        auto c = X(randInt, X.cSlice()); // pick a random point in our dataset
        centroids.put(i, centroids.cSlice(), c); // put it in our centroids matrix
    }

    return centroids;    
}


double kmeans::calculate_distance(const nc::NdArray<double>& point1, const nc::NdArray<double>& point2) {
    // calculate the euclidean distance (second norm) between two points
    return nc::norm<double>(point1 - point2)(0, 0);

}

nc::NdArray<nc::uint32> kmeans::assign_labels(nc::NdArray<double>& X, nc::NdArray<double>& clusters) {
    /*
    Assign all points to their nearest centroid.
    
    Parameters 
    ---------
    X: data on which KMeans clustering is being performed.
    clusters: array of size (n_clusters, X.shape().cols) which contains the centroid data.


    Returns 
    ----------
    labels: array of size (X.shape().rows, 1). The i-th element in labels is the cluster id that the i-th sample in X
    is assinged to.
    */

    int n_samples = X.shape().rows;
    int n_clusters = clusters.shape().rows;

    nc::NdArray<double> distances = nc::zeros<double>(n_samples, n_clusters); // (# points, # centroids)

    for (int i = 0; i < n_samples; i++) {
        // for each point
        for (int j = 0; j < n_clusters; j++) {
            // for each centroid

            double dist = calculate_distance(clusters(j, clusters.cSlice()), X(i, X.cSlice()));

            distances.put(i, j, dist);

        }
    }

    nc::NdArray<nc::uint32> labels = nc::argmin(distances, nc::Axis::COL).transpose();

    return labels;
}

nc::NdArray<double> kmeans::update_centroids(nc::NdArray<double>& X, nc::NdArray<nc::uint32> labels) {
    /*

    Move centroids to their new location
    
    Parameters 
    ---------
    X: data on which KMeans clustering is being performed.
    
    labels: array of size (X.shape().rows, 1). The i-th element in labels is the cluster id that the i-th sample in X
    is assinged to.


    Returns 
    ----------
    new_centroids: an array of size (n_clusters, X.shape().cols) containing the locations of the updated centroids.
    */

    int n_samples = X.shape().rows;
    int n_features = X.shape().cols;

    nc::NdArray<double> new_centroids = nc::zeros<double>(n_clusters, n_features);

    nc::NdArray<int> label_counts = nc::zeros<int>(n_clusters, 1);

    for (int i = 0; i < n_samples; i++) {
        // for each point

        auto point = X(i, X.cSlice());

        int l = labels(i, 0);

        label_counts.put(l, 0, label_counts(l, 0) + 1); // incrememnt the number of times we've seen label l

        auto _new_centroid = new_centroids(l, new_centroids.cSlice()); // grab the centroid corresponding to the label

        _new_centroid += point;

        new_centroids.put(l, new_centroids.cSlice(), _new_centroid);

    }

    new_centroids = new_centroids / label_counts.astype<double>();

    return new_centroids;
}

void kmeans::fit(nc::NdArray<double>& X) {
    /*
    Fitting function to perform k means clustering on input data X.

    KMeans algorithm is as follows: On each iteration do the following:
        1. move the centroids to the average location of all points assigned to it
            update_centroid_locations()
        2. recalculate the assignments
            assign_labels()
        3. move the centroids to their new average location
            move_centroids()
        4. check stopping criterion, if not, repeat
            check_stopping_criterion()


    Parameters 
    ---------
    X: Input array of shape (n_samples, n_features) on which to perform clustering. n_samples must be
    larger than the number of clusters requested on model instantiation.


    Returns 
    ---------
    Does not return anything. However, does set the is_fit flag to true, allowing 
    the user to access the centroids via the .get_centroids() method.


    Example
    ----------
    const int n_clusters = 3;
    const int max_iter = 10000; 
    const double tol = 0.001;

    kmeans k = kmeans(n_clusters, max_iter, tol);

    k.fit(X);
    nc::NdArray<double> centroids = k.get_centroids();
    */
         

    nc::NdArray<nc::uint32> labels;
    nc::NdArray<double> centroids;

    for (int i = 0; i < max_iter; i++) {
        centroids = initialize_centroids(X, n_clusters);

        labels = assign_labels(X, centroids);

        centroids = update_centroids(X, labels);

    }

    is_fit = true;

}


nc::NdArray<nc::uint32> kmeans::predict(nc::NdArray<double>& X) {
    /*
    Parameters 
    ---------
    X: Input array of shape (n_samples, n_features) on which to perform clustering. n_samples must be
    larger than the number of clusters requested on model instantiation.


    Returns 
    ----------
    labels: array of size (X.shape().rows, 1). The i-th element in labels is the cluster id that the i-th sample in X
    is assinged to.


    Example
    ----------
    const int n_clusters = 3;
    const int max_iter = 10000; 
    const double tol = 0.001;

    kmeans k = kmeans(n_clusters, max_iter, tol);

    k.fit(X_train);

    nc::NdArray<double> predicted_labels = k.predict(X_test);
    */


    if (!is_fit) {
        std::runtime_error("Error: Please call .fit() method before calling .predict() method.");
    }

    else if (X.shape().cols != centroids.shape().cols) {
        std::runtime_error("Error: Number of features in the data passed to .predict() method does not match number of features  in the data passed to .fit() method.");
    }

    nc::NdArray<nc::uint32> labels = assign_labels(X, centroids);

    return labels;
}


nc::NdArray<double> kmeans::get_centroids() {
    // getter to return the centroids after the model has been fit.
    if (!is_fit) {
        std::runtime_error("Error: Please call .fit() method before calling any getters.");
    }

    return centroids;
}