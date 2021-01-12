# Machine Learning Homework - Exoplanet Exploration

![exoplanets.jpg](Images/exoplanets.jpg)

This project explores different machine learning models to determine which may be best to identify exoplanents from NASA data.

This repository contains two folders named "data" and "code".  The data folder contains the csv data used to feed the models and the code folder contains four jupyter notebooks, one for each model.  To run these models you will need pandas, scikit-learn, gridsearch, and matplotlib installed on your computer. 

### Selecting The Models

For this homework I used the following classification models. 
    *   Decision Tree Classifier
    *   Random Forest Classifier
    *   SVC (a type of Support Vector Machine) 
    *   K Nearest Neighbor (KNN)

## Preparing The Models

I first determined the feature importances by running "Decision Tree" and "Random Forest" models.  The features that showed the most promise were: 'koi_fpflag_co', 'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_model_snr', 'koi_prad', 'koi_prad_err2', 'koi_duration_err2'.  Then I used the features to train the K Nearest Neighbors and SVC models while using GridSearchCV to hypertune the models and performance. Once I had the models, they were then saved in the "code" file in two different formats. 

### Preprocess the Data

* Preprocessed the dataset prior to fitting the model.
* Performed feature selection and removed unnecessary features.
* Used `MinMaxScaler` to scale the numerical data.
* Separated the data into training and testing data.

### Tune Model Parameters

* Used `GridSearch` to tune model parameters.
* Tuned and compared at least two different classifiers.

## Performance

## SVC

![svc.png](images/svc.jpg)
Model accuracy began at ~0.64. Using only important features identified by the Random Forest model, it improved to 0.887. This was achieved using a train-test split of 80-20 and the default SVC kernel (rbf). Using linear and poly kernel settings were resource and time intensive.
SVC appears to be the better choice for potential planet classification, having especially high precision for identifying false positives.

## K Nearest Neighbors

![knn.png](images/KNN.jpg)
Achieved model accuracy of 0.866 with k=15, using important features identified by the Random Forest model.  The K Nearest Neighbors model is much faster than SVC, which may make it a better model to use on larger datasets. However, for this dataset (which is relatively small), speed is not as much of an issue.  While the model accuracy is slightly higher overall, the precision for classifying different planet types indicated in the classification_report is lower.


### Reporting

* As you can see in the descriptions of the two models that I ran through, the SVC model is the best and most accurate model to use.  Even though this model takes a little longer to run than the KNN model, it yields better results.  I only used GridSearchCV on the SVC model as it showed the most accuracy.  I feel that the model could be used to predict new exoplanets with a high degree of certainity if more fine tuned models and interations were ran in conjuction to determine a better degree of accuracy. 

n make based on your model (is your model good enough to predict new exoplanets? Why or why not? What would make your model be better at predicting new exoplanets?).

- - -

## Resources

* [Exoplanet Data Source](https://www.kaggle.com/nasa/kepler-exoplanet-search-results)

* [Scikit-Learn Tutorial Part 1](https://www.youtube.com/watch?v=4PXAztQtoTg)

* [Scikit-Learn Tutorial Part 2](https://www.youtube.com/watch?v=gK43gtGh49o&t=5858s)

* [Grid Search](https://scikit-learn.org/stable/modules/grid_search.html)

