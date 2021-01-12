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

## Performance

## SVC

![svc.png](images/svc.jpg)
Model accuracy began at ~0.64. Using only important features identified by the Random Forest model, it improved to 0.887. This was achieved using a train-test split of 80-20 and the default SVC kernel (rbf). Using linear and poly kernel settings were resource and time intensive.
SVC appears to be the better choice for potential planet classification, having especially high precision for identifying false positives.

## K Nearest Neighbors

![knn.png](images/knn.jpg)


## Instructions

### Preprocess the Data

* Preprocess the dataset prior to fitting the model.
* Perform feature selection and remove unnecessary features.
* Use `MinMaxScaler` to scale the numerical data.
* Separate the data into training and testing data.

### Tune Model Parameters

* Use `GridSearch` to tune model parameters.
* Tune and compare at least two different classifiers.

### Reporting

* Create a README that reports a comparison of each model's performance as well as a summary about your findings and any assumptions you can make based on your model (is your model good enough to predict new exoplanets? Why or why not? What would make your model be better at predicting new exoplanets?).

- - -

## Resources

* [Exoplanet Data Source](https://www.kaggle.com/nasa/kepler-exoplanet-search-results)

* [Scikit-Learn Tutorial Part 1](https://www.youtube.com/watch?v=4PXAztQtoTg)

* [Scikit-Learn Tutorial Part 2](https://www.youtube.com/watch?v=gK43gtGh49o&t=5858s)

* [Grid Search](https://scikit-learn.org/stable/modules/grid_search.html)

- - -

## Hints and Considerations

* Start by cleaning the data, removing unnecessary columns, and scaling the data.

* Not all variables are significant be sure to remove any insignificant variables.

* Make sure your `sklearn` package is up to date.

* Try a simple model first, and then tune the model using `GridSearch`.

* When hyper-parameter tuning, some models have parameters that depend on each other, and certain combinations will not create a valid model. Be sure to read through any warning messages and check the documentation

- - -

## Submission

* Create a Jupyter Notebook for each model and host the notebooks on GitHub.

* Create a file for your best model and push to GitHub

* Include a README.md file that summarizes your assumptions and findings.

* Submit the link to your GitHub project to Bootcamp Spot.

* Ensure your repository has regular commits (i.e. 20+ commits) and a thorough README.md file

##### © 2020 Trilogy Education Services, a 2U, Inc. brand. All Rights Reserved.
