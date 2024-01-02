# ML_CPP Documentation
## Table of Contents
1. [Preprocessing](#preprocessing)
   
	1.2 [Data Ingestion](#data_ingestion)
   
	1.2.1 [Read CSV](#read_csv)
   
	1.2.2 [Dataset](#dataset)
   
	1.3 [Scalers](#scalers)

	1.3.1 [Standard Scaler](#standard_scaler)

	1.3.2 [Min Max Scaler](#minmax_scaler)

3. [Models](#models)
   
	2.1 [Classification](#classification)
   
	2.1.1 [Logistic Regression](#logistic_regression)
   
	2.1.2 [Random Forest Classifier](#randomforest_classifier)
   
	2.2 [Regression](#regression)
   
	2.2.1 [Linear Regression](#linear_regression)
   
	2.3 [Clustering](#clustering)
   
	2.3.1 [K Means Clustering](#kmeans)
   
5. [Evaluation](#evaluation)
 
	3.1 [Classification Metrics](#classification_metrics)
   
	3.1.1 [Accuracy Score](#accuracy_score)
   
	3.1.2 [F1 Score](#f1_score)
   
	3.1.3 [Precision Score](#precision_score)
   
	3.1.4 [Recall Score](#recall_score)
   
	3.1.5 [confusion_matrix](#confusion_matrix)
   
	3.2 [Regression Metrics](#regression_metrics)
   
	3.2.1 [Max Error](#max_error)
   
	3.2.2 [Mean Absolute Error](#mean_absolute_error)
   
	3.2.3 [Mean Squared Error](#mean_squared_error)

		
## Dependencies
[NumCpp](https://github.com/dpilger26/NumCpp): Library used to perform the various linear algebra and other mathematical functions.
[Catch2](https://github.com/catchorg/Catch2): Library used for testing.

## Preprocessing <a name="preprocessing"></a>
Preprocessing functions are used to load and prepare data for use in a machine learning model.
### Data Ingestion <a name="data_ingestion"></a>
#### Read CSV <a name="read_csv"></a>
The `read_csv` function can be used to load the contents of a  `.csv` file into a NumCpp array. The intention is for the user to then load this array into an instance of the `dataset` class.

    nc::NdArray<double> read_csv(const  std::string& filepath, bool skip_header);

#### Dataset <a name="dataset"></a>
The dataset class is used to load and split data into test and train sets. The intention is to load data from a `.csv` using `read_csv`, and then load the resulting matrix into an instance of the `dataset` class.

Constructor:

    dataset(nc::NdArray<double> data);

> **Parameters**
> - data: NumCpp array of doubles containing the data to be processed.
> 
> **Returns**
> - matrix: NumCpp array of doubles containing the data in the .csv file.
---

The `dataset` class exposes the following methods:    



    void train_test_split(double train_size, bool shuffle);

> Splits the array loaded in the constructor into X, y, X_train, y_train, X_test, and y_test, and saves them into attributes of this class, accessible via getters. Note that the labels/targets are taken from the final (right-most) column of the matrix.
> 
> **Parameters:**
> - train_size: the size of the train set. Must be between 0 and 1. The size of the test set is equal to `1.0 - train_set`.
> - shuffle: boolean indicating whether or not to perform a row-wise shuffle of the data before performing the splitting operation. Shuffling is done via the [Fisher-Yates Algorithm](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle).

---
    nc::NdArray<double> get_X();

> Getter to return all input data from train and test sets. `.train_test_split(...)` must be called beforehand.
> 
> **Returns**
> X: NumCpp array containing all input data from train and test sets.
---
    nc::NdArray<double> get_X_train();

> Getter to return all input data from train set. `.train_test_split(...)` must be called beforehand.
> 
> **Returns**
> X_train: NumCpp array containing all input data from train set.
---

    nc::NdArray<double> get_X_test();

> Getter to return all input data from test set. `.train_test_split(...)` must be called beforehand.
> 
> **Returns**
> X_test: NumCpp array containing all input data from test set.
---
    nc::NdArray<double> get_y();

> Getter to return all label/target data from train and test sets. `.train_test_split(...)` must be called beforehand.
> 
> **Returns**
> y: NumCpp array containing all label/target data from train and test sets.
---

    nc::NdArray<double> get_y_train();

> Getter to return all label/target data from train set. `.train_test_split(...)` must be called beforehand.
> 
> **Returns**
> y_train: NumCpp array containing all label/target data from train set.

    nc::NdArray<double> get_y_test();

> Getter to return all label/target data from test set. `.train_test_split(...)` must be called beforehand.
> 
> **Returns**
> y_test: NumCpp array containing all label/target data from test set.


### Scalers <a name="scalers"></a>
Scalers are used to scaled input data to potentially increase numerical stability and computation time. 
#### Standard Scaler <a name="standard_scaler"></a>
Standard scaler scales data to zero mean and unit variance on a feature-by-feature basis.

Constructor:

    standard_scaler(bool with_mean = true, bool with_std = true);

> **Parameters:**
> - with_mean: boolean variable, if set to true, scaler will scale each feature in the data to a mean of 0.
> with_std: boolean variable, if set to true, scaler will scale each feature in the data to a variance of 1.

The `standard_scaler` class exposes the following methods:    

    void fit(nc::NdArray<double>& X);

> Function to fit scaler to the data. Must be called before the `.transform(...)` method may be called. 
> 
> **Parameters:**
> - X: NumCpp array of doubles containing training data of shape (# samples, # features). May not contain any NaN values.

---

    nc::NdArray<double> transform(nc::NdArray<double>& X);

> Transforms the data based based on the standard deviations and means calculated during the `.fit(...)` method. `.fit(...)` or `.fit_transform(...)` method must be called before this may be called.
> **Parameters:**
> - X: NumCpp array of doubles containing data of shape (# samples, # features) on which to perform the transformation.
> 
> **Returns:**
> - X_scaled: NumCpp array of doubles of shape (# samples, # features) containing the transformed data.


---

    nc::NdArray<double> fit_transform(nc::NdArray<double>& X);

> Fits the scaler on the input data X and then transforms it. Equivalent to calling `.fit(...)` and then `.transform(...)`.
> **Parameters**
> - X: NumCpp array of doubles of shape (# samples, # features) on which to fit the scaler and then perform transformation operation.
> **Returns**
> - X_scaled: NumCpp array of doubles of shape (# samples, # features) containing the transformed data.

---

    nc::NdArray<double> inverse_transform(nc::NdArray<double>& X);

> Undoes the `.transform(...)` operation. `.fit(...)` or `.fit_transform(...)` method must be called before this may be called.
> **Parameters**
> - X: NumCpp array of doubles of shape (# samples, # features) on which to fit the scaler and then perform transformation operation.
> **Returns**
> - X_scaled: NumCpp array of doubles of shape (# samples, # features) containing the transformed data.

---
    nc::NdArray<double> get_means();

> Returns the feature means calculated during the fitting process. `.fit(...)` or `.fit_transform(...)` method must be called before this may be called.
> **Returns**
> - means: NumCpp array of doubles of shape (1, # features) containing means of the features calculated during fitting.
---
    nc::NdArray<double> get_stds();

> Returns the feature standard deviations calculated during the fitting process. `.fit(...)` or `.fit_transform(...)` method must be called before this may be called.
> **Returns**
> - stds: NumCpp array of doubles of shape (1, # features) containing standard deviations of the features calculated during fitting.

---

#### Minmax Scaler <a name="minmax_scaler"></a>
The `minmax_scaler` transforms the data by scaling the features to a provided range `(feature_min, feature_max)` 

Constructor:

    minmax_scaler(double feature_min = 0.0, double feature_max = 1.0);

> **Parameters:**
> - feature_min: minimum value to scale values to. All data will be greater than or equal to this value.
> feature_max:maximum vlaue to scale values to. All data will be less than or equal to this value.

The `minmax_scaler` class exposes the following methods:    

    void fit(nc::NdArray<double>& X);

> Function to fit scaler to the data. Must be called before the `.transform(...)` method may be called. 
> 
> **Parameters:**
> - X: NumCpp array of doubles containing training data of shape (# samples, # features). May not contain any NaN values.

---

    nc::NdArray<double> transform(nc::NdArray<double>& X);

> Method to transform the data based on the feature maximums and minimums calculated during the `.fit(...)` method. `.fit(...)` or `.fit_transform(...)` method must be called before this may be called.
> **Parameters:**
> - X: NumCpp array of doubles containing data of shape (# samples, # features) on which to perform the transformation.
> 
> **Returns:**
> - X_scaled: NumCpp array of doubles of shape (# samples, # features) containing the transformed data.
---

    nc::NdArray<double> fit_transform(nc::NdArray<double>& X);

> Fits the scaler on the input data X and then transforms it. Equivalent to calling `.fit(...)` and then `.transform(...)`.
> **Parameters**
> - X: NumCpp array of doubles of shape (# samples, # features) on which to fit the scaler and then perform transformation operation.
> **Returns**
> - X_scaled: NumCpp array of doubles of shape (# samples, # features) containing the transformed data.

---

    nc::NdArray<double> inverse_transform(nc::NdArray<double>& X);

> Undoes the `.transform(...)` operation. `.fit(...)` or `.fit_transform(...)` method must be called before this may be called.
> **Parameters**
> - X: NumCpp array of doubles of shape (# samples, # features) on which to fit the scaler and then perform transformation operation.
> **Returns**
> - X_scaled: NumCpp array of doubles of shape (# samples, # features) containing the transformed data.

---
    nc::NdArray<double> get_min_vals();

> Returns the feature minimums calculated during the fitting process. `.fit(...)` or `.fit_transform(...)` method must be called before this may be called.
> **Returns**
> - min_vals: NumCpp array of doubles of shape (1, # features) containing minimums of the features calculated during fitting.
---
    nc::NdArray<double> get_max_vals();

> Returns the feature maximums calculated during the fitting process. `.fit(...)` or `.fit_transform(...)` method must be called before this may be called.
> **Returns**
> - max_vals: NumCpp array of doubles of shape (1, # features) containing maximums of the features calculated during fitting.

---

## Models <a name="models"></a>
### Classification <a name="classification"></a>
Classification type models are used to predict discrete values (i.e. categories) to which a sample belongs.

#### Logistic Regression <a name="logistic_regression"></a>
Implementation of a [Logistic Regression model](https://en.wikipedia.org/wiki/Logistic_regression) with regularisation support using a cross-entropy loss function. Supports multi-class classification via a one-vs-rest scheme.

Constructor:

    logistic_regression(const std::string penalty = "l2", const double reg_strength = 0.1, const int max_iters = 1000, const double lr = 0.01, const double tol = 0.0001, const int init_mode = 1);

> **Parameters:**
> - penalty: Type of regularisation to use. "l2" is the default option, with "l1" also being supported. Otherwise, no regularisation will be applied.
> - reg_strength: Strength of regularisation. Larger value indicates stronger regularisation.
> - max_iters: maximum iterations for with the model will train. 
> - lr: Float value representing the learning rate during training. Larger values indicate a stronger learning rate.
> - tol: Tolerance used to determine early stopping criterion. If model loss changes by less than the tol value between iterations, early stopping will occur.
> - init_mode: Mode in which to initialise the model weights. If set to 1, weights will be initialised to random values perturbed around 0.
> Otherwise, all weights will be initialised to 0.

The `logistic_regression` class exposes the following methods:    

    void fit(nc::NdArray<double>& X, nc::NdArray<double>& y, bool verbose);

> Main fitting function. Must be called before the `.predict(...)` method may be called. 
> 
> **Parameters:**
> - X: NumCpp array of doubles containing training data of shape (# samples, # features). May not contain any NaN values.
> - y: NumCpp array of doubles containing training labels. May be of the shape (# samples, 1) or (1, # samples). May not contain any NaN values.
> - verbose: boolean value indicating whether or not to output information during training. If set to true, the objective cost every 100 iterations will be written to the standard output.


---

    nc::NdArray<double> predict(nc::NdArray<double>& X);

> Method to perform predictions on an input.
> **Parameters:**
> - X: NumCpp array of doubles containing data of shape (# samples, # features) on which to perform prediction. May not contain any NaN
> values.
> 
> **Returns:**
> - NumCpp array of doubles of shape (# samples, 1) containing the predicted values.


---

    const nc::NdArray<double> get_weights() const;

> Method to return the weights of the model after the `.fit(...)` method has been called.

---

    const nc::NdArray<double> get_bias() const;

> Method to return the bias of the model after the `.fit(...)` method has been called.


#### Random Forest Classifier <a name="randomforest_classifier"></a>
Implementation of a [Random Forest Classifier](https://en.wikipedia.org/wiki/Random_forest) . Supports multi-class classification.

Constructor:

    randomforest_classifier(const int n_estimators = 10, const int max_depth = 5, const int  min_samples_split = 5, int max_features = -1);

> **Parameters:**
> - n_estimators: Number of decision trees to grow in the random forest.
> - max_depth: Maximum depth to which each decision tree will be grown.
> - min_samples_split: The minimum number of samples in a decision tree node required to continue splitting.
> - max_features: Maximum number of features to randomly sample for each decision tree. Must be a positive integer value greater than 0. If left as the default -1, the square root of the total number of features, rounded to the nearest integer, will be used.

The `randomforest_classifier` class exposes the following methods:    

    void fit(nc::NdArray<double>&  X, nc::NdArray<double>&  y, bool verbose);

> Main fitting function. Must be called before the `.predict(...)` method may be called. 
> 
> **Parameters:**
> - X: NumCpp array of doubles containing training data of shape (# samples, # features). May not contain any NaN values.
> - y: NumCpp array of doubles containing training labels. May be of the shape (# samples, 1) or (1, # samples). May not contain any NaN values. 
> - verbose: boolean value indicating whether or not to output information during training. If set to true, the model will write to the standard output each decision tree's out-of-bag score.


---

    nc::NdArray<double> predict(nc::NdArray<double>&  X);

> Method to perform predictions on an input.
> **Parameters:**
> - X: NumCpp array of doubles containing data of shape (# samples, # features) on which to perform prediction. May not contain any NaN values.
> 
> **Returns:**
> - NumCpp array of doubles of shape (# samples, 1) containing the predicted values.


---

### Regression <a name="regression"></a>
Regression type models predict a continuous value associated with a sample.

#### Linear Regression <a name="linear_regression"></a>
Implementation of a [Linear Regression model](https://en.wikipedia.org/wiki/Linear_regression) with regularisation support.

Constructor:

    linear_regression(const std::string penalty = "l2", const double reg_strength = 0.1, const int max_iters = 1000, const double lr = 0.01, const double tol = 0.0001, const int init_mode = 1);

> **Parameters:**
> - penalty: Type of regularisation to use. "l2" is the default option, with "l1" also being supported. Otherwise, no regularisation will be applied.
> - reg_strength: Strength of regularisation. Larger value indicates stronger regularisation.
> - max_iters: maximum iterations for with the model will train. 
> - lr: Float value representing the learning rate during training. Larger values indicate a stronger learning rate.
> - tol: Tolerance used to determine early stopping criterion. If model loss changes by less than the tol value between iterations, early stopping will occur.
> - init_mode: Mode in which to initialise the model weights. If set to 1, weights will be initialised to random values perturbed around 0. Otherwise, all weights will be initialised to 0.

The `linear_regression` class exposes the following methods:    

    void fit(nc::NdArray<double>& X, nc::NdArray<double>& y, bool verbose);

> Main fitting function. Must be called before the `.predict(...)` method may be called. 
> 
> **Parameters:**
> - X: NumCpp array of doubles containing training data of shape (# samples, # features). May not contain any NaN values.
> - y: NumCpp array of doubles containing training labels. May be of the shape (# samples, 1) or (1, # samples). May not contain any NaN values.
> - verbose: boolean value indicating whether or not to output information during training. If set to true, the objective cost at each iteration will be written to the standard output.


---

    nc::NdArray<double> predict(nc::NdArray<double>& X);

> Method to perform predictions on an input.
> **Parameters:**
> - X: NumCpp array of doubles containing data of shape (# samples, # features) on which to perform prediction. May not contain any NaN values.
> 
> **Returns:**
> - NumCpp array of doubles of shape (# samples, 1) containing the predicted values.


---

    const  nc::NdArray<double> get_weights() const;

> Method to return the weights of the model after the `.fit(...)` method has been called.

---

    const  nc::NdArray<double> get_bias() const;

> Method to return the bias of the model after the `.fit(...)` method has been called.

---

## Clustering <a name="clustering"></a>

Clustering algorithms group different samples together based on their features.

#### K Means Clustering <a name="kmeans"></a>
Implementation of a [K Means Clustering model](https://en.wikipedia.org/wiki/K-means_clustering).

Constructor:

    kmeans(const int n_clusters, const int max_iter, const double tol);

> **Parameters:**
> - n_clusters: Number of clusters in which to segment data. The "K" in "K-Means".
> - max_iters: maximum iterations for with the model will train. 
> - tol: Tolerance used to determine early stopping criterion. If the sum of the absolute difference in centroid coordinates changes by less than tol between iterations, the training will stop.

The `kmeans` class exposes the following methods:    

    void fit(nc::NdArray<double>& X, bool verbose);

> Main fitting function. Must be called before the `.predict(...)` method may be called. 
> 
> **Parameters:**
> - X: NumCpp array of doubles containing training data of shape (# samples, # features). May not contain any NaN values.
> - verbose: boolean value indicating whether or not to output information during training. If set to true, the sum of the absolute values of centroid movement on each iteration will be printed to the standard output.


---

    nc::NdArray<double> predict(nc::NdArray<double>& X);

> Method to perform predictions on an input.
> **Parameters:**
> - X: NumCpp array of doubles containing data of shape (# samples, # features) on which to perform prediction. May not contain any NaN values.
> 
> **Returns:**
> - NumCpp array of doubles of shape (# samples, 1) containing the predicted values.


---

    const nc::NdArray<double> get_centroids() const;

> Method to return the centroids of the model after the `.fit(...)` method has been called.

---

## Evaluation <a name="evaluation"></a>
The following functions are used to evaluate model performance for both classification type problems, and regression type problems.
### Classification Metrics <a name="classification_metrics"></a>
Classification metrics are used to compare the performance of classifiers.
#### Accuracy Score <a name="accuracy_score"></a>
Calculate the accuracy score of a set of predictions. Accuracy is defined as the number of correct classifications / the total number of classifications.

    double accuracy_score(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred);
    
> **Parameters:**
> - y_true: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the ground-truth values against which to evaluate the predictions. May not contain any NaN values.
> - y_pred: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the predictions to evaluate. May not contain any NaN values.
> 
> **Returns:**
> - score: A double representing the accuracy of the predictions. 
---
#### F1 Score <a name="f1_score"></a>
Calculate f1 scores for a set of predictions. F1 is defined as: `2 * (precision + recall) / (precision + recall)` where `precision` and `recall` are the precision score and recall score respectively. F1 is a contrived metric and therefore doesn't have an intuitive meaning, but rather is the harmonic mean of the precision and recall. The best f1 score is 1, and the worst is 0.

    nc::NdArray<double> f1_score(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred);
    
> **Parameters:**
> - y_true: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the ground-truth values against which to evaluate the predictions. May not contain any NaN values.
> - y_pred: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the predictions to evaluate. May not contain any NaN values.
> 
> **Returns:**
> - score: NumCpp array of doubles of size (1, # classes) representing the f1 score of the predictions for each class. 
---
#### Precision Score <a name="precision_score"></a>
Calculate precision scores for a set of predictions. Precision is defined as: `tp / (tp + fp)` where `tp` is the number of true positives, and `fp` is the number of false positives. Intuitively the precision of a classifier is its ability to not label a positive samples as negative. The best precision value is 1, and the worst is 0.

    nc::NdArray<double> precision_score(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred);
    
> **Parameters:**
> - y_true: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the ground-truth values against which to evaluate the predictions. May not contain any NaN values.
> - y_pred: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the predictions to evaluate. May not contain any NaN values.
> 
> **Returns:**
> - score: NumCpp array of doubles of size (1, # classes) representing the precision score of the predictions for each class. 
---
#### Recall Score <a name="recall_score"></a>
Calculate recall scores for a set of predictions. Recall is defined as: `tp / (tp + fn)` where `tp` is the number of true positives, and `fn` is the number of false negatives. Intuitively recall represents a classifier's ability to find the positives. The best recall value is 1, and the worst is 0.

    nc::NdArray<double> recall_score(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred);
    
> **Parameters:**
> - y_true: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the ground-truth values against which to evaluate the predictions. May not contain any NaN values.
> - y_pred: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the predictions to evaluate. May not contain any NaN values.
> 
> **Returns:**
> - score: NumCpp array of doubles of size (1, # classes) representing the recall score of the predictions for each class. 
#### Confusion Matrix <a name="confusion_matrix"></a>
Returns an (# classes x # classes) [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) of the predictions. Rows represent the true labels, and columns the predicted labels. For confusion matrix C, the value at the i-th row and j-th column, C(i, j) is the number of observations in group i, predicted to be in group j. 

    nc::NdArray<double> confusion_matrix(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred);

    
> **Parameters:**
> - y_true: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the ground-truth values against which to evaluate the predictions. May not contain any NaN values.
> - y_pred: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the predictions to evaluate. May not contain any NaN values.
> 
> **Returns:**
> - c_matrix: NumCpp array of doubles of shape (# classes x # classes). 

### Regression Metrics <a name="regression_metrics"></a>
Regression metrics are used to evaluate the performance of regressors.
#### Max Error <a name="max_error"></a>
Calculate the maximum residual error between the target values and the estimated values.

    double max_error(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred);
    
> **Parameters:**
> - y_true: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the ground-truth values against which to evaluate the predictions. May not contain any NaN values.
> - y_pred: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the predictions to evaluate. May not contain any NaN values.
> 
> **Returns:**
> - score: A double representing the maximum error.
#### Mean Absolute Error <a name="mean_absolute_error"></a>
Calculate the average absolute residual error between the target values and the estimated values.

    double mean_absolute_error(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred);
    
> **Parameters:**
> - y_true: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the ground-truth values against which to evaluate the predictions. May not contain any NaN values.
> - y_pred: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the predictions to evaluate. May not contain any NaN values.
> 
> **Returns:**
> - score: A double representing the mean absolute error.

#### Mean Squared Error <a name="mean_squared_error"></a>
Calculate the average squared residual error between the target values and the estimated values.

    double mean_squared_error(nc::NdArray<double>& y_true, nc::NdArray<double>& y_pred);
    
> **Parameters:**
> - y_true: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the ground-truth values against which to evaluate the predictions. May not contain any NaN values.
> - y_pred: NumCpp array of doubles containing data of shape (# samples, 1) or (1, # samples) containing the predictions to evaluate. May not contain any NaN values.
> 
> **Returns:**
> - score: A double representing the mean squared error.

