from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import joblib
import os

# load database function
def load_data(file_name):
    return np.loadtxt("datasets/tictac_" + file_name + ".txt")

### DATASETS
final_bc = load_data("final")
multi_bc = load_data("multi")
single_bc = load_data("single")


def evaluate_classifier(classifier, data, dn):

    print("Evaluating classifer for " + dn + " dataset")

    # Extract features (X) and labels (y)
    X = data[:, :9]
    y = data[:, 9]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # train
    classifier.fit(X_train, y_train)

    # Predict the optimal moves for the entire dataset
    y_pred = classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Perform k-fold cross-validation
    cv_scores = cross_val_score(classifier, X, y, cv=10)

    # Print results
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Cross-validation scores:", cv_scores)
    print("Mean accuracy:", np.mean(cv_scores))

    return classifier

def evaluate_regressor(regressor, data, dataset_name):
    print("Evaluating regressor for " + dataset_name + " dataset")

    # Extract features (X) and labels (y)
    X = data[:, :9]
    y = data[:, 9]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the regressor
    regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = regressor.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # Perform k-fold cross-validation
    cv_mse_scores = -cross_val_score(regressor, X, y, cv=10, scoring='neg_mean_squared_error')

    # Print results
    print("Mean Squared Error:", mse)
    print("Cross-validation MSE scores:", cv_mse_scores)
    print("Mean Cross-validation MSE:", np.mean(cv_mse_scores))
    print() 
    return regressor

# Make folder called models
models_directory = 'models'
if not os.path.exists(models_directory):
    os.makedirs(models_directory)
## CLASSFIERS

# Evaluate KNN classifier
print("Evaluate KNN classifier\n")
knn_classifier_sbc = evaluate_classifier(KNeighborsClassifier(n_neighbors=5), single_bc, "tictac_single")
knn_classifier_mbc = evaluate_classifier(KNeighborsClassifier(n_neighbors=5), multi_bc, "tictac_final")
knn_classifier_fbc = evaluate_classifier(KNeighborsClassifier(n_neighbors=5), final_bc, "tictac_multi")
joblib.dump(knn_classifier_sbc, os.path.join(models_directory, 'knn_classifier_sbc.pkl'))
joblib.dump(knn_classifier_mbc, os.path.join(models_directory, 'knn_classifier_mbc.pkl'))
joblib.dump(knn_classifier_fbc, os.path.join(models_directory, 'knn_classifier_fbc.pkl'))
print() 

# Evaluate MLP classifier
print("Evaluate MLP classifier\n") 
mlp_classifier_sbc = evaluate_classifier(MLPClassifier(max_iter=10000), single_bc, "tictac_single")
mlp_classifier_mbc = evaluate_classifier(MLPClassifier(max_iter=10000), multi_bc, "tictac_final")
mlp_classifier_fbc = evaluate_classifier(MLPClassifier(max_iter=10000), final_bc, "tictac_multi")
joblib.dump(mlp_classifier_sbc, os.path.join(models_directory,'mlp_classifier_sbc.pkl'))
joblib.dump(mlp_classifier_mbc, os.path.join(models_directory, 'mlp_classifier_mbc.pkl'))
joblib.dump(mlp_classifier_fbc, os.path.join(models_directory,'mlp_classifier_fbc.pkl'))
print() 

# Evaluate SVM classifier
print("Evaluate SVM classifier\n")
svm_classifier_sbc = evaluate_classifier(LinearSVC(dual=True), single_bc, "tictac_single")
svm_classifier_mbc = evaluate_classifier(LinearSVC(dual=True), multi_bc, "tictac_final")
svm_classifier_fbc = evaluate_classifier(LinearSVC(dual=True), final_bc, "tictac_multi")
joblib.dump(svm_classifier_sbc, os.path.join(models_directory,'svm_classifier_sbc.pkl'))
joblib.dump(svm_classifier_mbc, os.path.join(models_directory,'svm_classifier_mbc.pkl'))
joblib.dump(svm_classifier_fbc, os.path.join(models_directory,'svm_classifier_fbc.pkl'))
print() 

## REGRESSORS
print("Evaluate KNN regressor\n")
knn_regressor = evaluate_regressor(KNeighborsRegressor(), multi_bc, "multi_bc")
print("Evaluate Linear regressor\n")
linear_regressor = evaluate_regressor(LinearRegression(), multi_bc, "multi_bc")
print("Evaluate MLP regressor\n")
mlp_regressor = evaluate_regressor(MLPRegressor(max_iter=10000), multi_bc, "multi_bc")
print() 
joblib.dump(knn_regressor, os.path.join(models_directory,'knn_regressor.pkl'))
joblib.dump(linear_regressor, os.path.join(models_directory,'linear_regressor.pkl'))
joblib.dump(mlp_regressor, os.path.join(models_directory,'mlp_regressor.pkl'))



