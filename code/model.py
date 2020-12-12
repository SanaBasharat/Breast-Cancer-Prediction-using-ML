from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np #Python library for operations on  multi-dimensional arrays and matrices
import pandas as pd #Python library to manipulate datasets
import matplotlib.pyplot as plt #generates graphs
from sklearn.model_selection import train_test_split
import pickle
# import requests
# import json
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
import lime
import lime.lime_tabular

dataset = pd.read_csv('dataset.csv') #loading data
X = dataset.iloc[:, :-1].values #separating features row
y = dataset.iloc[:, 5].values  #separating classifier column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) #splitting data
# 80% for training, 20% for testing

regressor = LogisticRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print("\nLOGISTIC REGRESSON: ")
print(classification_report(y_test,y_pred))
confusion_matrix_regressor = confusion_matrix(y_test, y_pred)
print(confusion_matrix_regressor)
skplt.metrics.plot_confusion_matrix(y_test, y_pred,figsize=(8,8))
plt.title('Confusion matrix - Logistic Regression')

randforest = RandomForestClassifier()
randforest.fit(X_train, y_train)
y_pred = randforest.predict(X_test)

print("\nRANDOM FOREST: ")
print(classification_report(y_test,y_pred))
confusion_matrix_randforest = confusion_matrix(y_test, y_pred)
print(confusion_matrix_randforest)
skplt.metrics.plot_confusion_matrix(y_test, y_pred,figsize=(8,8))
plt.title('Confusion matrix - Random Forest')

kmeans = KNeighborsClassifier()
kmeans.fit(X_train, y_train)
y_pred = kmeans.predict(X_test)

print ("\nK NEAREST NEIGHBOR: ")
print(classification_report(y_test,y_pred))
confusion_matrix_kmeans = confusion_matrix(y_test, y_pred)
print(confusion_matrix_kmeans)
skplt.metrics.plot_confusion_matrix(y_test, y_pred,figsize=(8,8))
plt.title('Confusion matrix - K Nearest Neighbor')
plt.show()

pickle.dump(regressor, open('model.pkl','wb'))

#model = pickle.load(open('model.pkl','rb'))
print("\nTesting a sample, with values 14.69, 13.98, 98.22, 656.1 and 0.1031: ")
print("Linear Regression: ")
print(regressor.predict([[14.69, 13.98, 98.22, 656.1, 0.1031]]))  #testing a sample
print("Random Forest")
print(randforest.predict([[14.69, 13.98, 98.22, 656.1, 0.1031]]))  #testing a sample
print("K Nearest Neighbor")
print(kmeans.predict([[14.69, 13.98, 98.22, 656.1, 0.1031]]))  #testing a sample

predict_fn_regressor = lambda x: regressor.predict_proba(x).astype(float)
predict_fn_randforest = lambda x: randforest.predict_proba(x).astype(float)
predict_fn_kmeans = lambda x: kmeans.predict_proba(x).astype(float)

feature_names = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
class_names = ['0','1']

explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names, discretize_continuous=True)

#Pick the observation/sample for which an explanation is required
sample = 10

# The Explainer Instance
exp_regressor = explainer.explain_instance(X_test[sample], predict_fn_regressor, num_features=5, top_labels=1)
exp_randforest = explainer.explain_instance(X_test[sample], predict_fn_randforest, num_features=5, top_labels=1)
exp_kmeans = explainer.explain_instance(X_test[sample], predict_fn_kmeans, num_features=5, top_labels=1)

# exp_regressor.show_in_notebook(show_all=False)
exp_regressor.save_to_file('regressor.html')

# exp_randforest.show_in_notebook(show_all=False)
exp_randforest.save_to_file('randforest.html')

# exp_kmeans.show_in_notebook(show_all=False)
exp_kmeans.save_to_file('kmeans.html')
