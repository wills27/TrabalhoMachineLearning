#Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

#Modelling
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, plot_roc_curve


x_train = pd.read_csv("x_train_data.csv")
x_test = pd.read_csv("x_test_data.csv")
y_train = pd.read_csv("y_train_data.csv")
y_test = pd.read_csv("y_test_data.csv")
train_data = pd.read_csv("wine-quality.csv", delimiter=";")

#K=21 has lowest error rate
#Model Fit
classifier2 = KNeighborsClassifier(n_neighbors= 21, metric = 'manhattan', p = 2,weights='uniform')
classifier2.fit(x_train,y_train)

#Predicting the ouput from input data (x_train) and (y_train) 
y_pred1 = classifier2.predict(x_train)
y_pred2 = classifier2.predict(x_test)

print("Accuracy score of train data set:",accuracy_score(y_train, y_pred1))
print("Accuracy score of test data set:",accuracy_score(y_test, y_pred2))

#Se o valor de qualidade for menor ou igual a 5, então ele vai ficar na classe 0
#Se o valor de qualidade for maior que 5, então ele vai ficar na classe 1
train_data['quality'] = np.where(train_data['quality'] > 5, 1, 0)
train_data['quality'].value_counts()

#Assigning dataframe to list of array values
X = train_data.drop(['quality'], axis = 1).values
y = train_data['quality'].values

k = range(1,50,2)
testing_accuracy = []
training_accuracy = []
score = 0
#Fitting the model
for i in k:
    knn = KNeighborsClassifier(n_neighbors = i)
    pipe_knn = Pipeline([('scale', MinMaxScaler()), ('knn', knn)])
    pipe_knn.fit(x_train, y_train)
    
    y_pred_train = pipe_knn.predict(x_train)
    training_accuracy.append(accuracy_score(y_train, y_pred_train))
    
    y_pred_test = pipe_knn.predict(x_test)
    acc_score = accuracy_score(y_test,y_pred_test)
    testing_accuracy.append(acc_score)
    
    if score < acc_score:
        score = acc_score
        best_k = i
        
print('Best Accuracy Score', score, 'Best K-Score', best_k)