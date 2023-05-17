'''Implement the KNN IRIS Image classification based on a given set ot training samples.Read the data from the .csv file
K-nearest neighbour: 
1)can be used for both classification and regression
2)KNN fails in the supervised learning family of algorithms
3)informally,means that we are given a labelled dataset.
STEPS
1)choose the number k of neighbours
2)take the k nearest neighbours of the new data point,according to your distance metric
3)among the k neighbours,count the number of data points to wach category
4)assign the new data point.'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
iris=pd.read_csv('/media/pain/DATA/MIET/SEM 6TH AI with Computer Vision/iris.csv')
iris.tail()
print(iris.shape)
iris['Species'].value_counts()
print(iris.columns)
print(iris.info())

X=iris.iloc[:,:4]
print(X.head())
y=iris.iloc[:,:-1]
print(y.head)
X=preprocessing.StandardScaler().fit_transform(X)
print(X[0:4])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3, random_state=1)
print(y_test.shape)
knnmodel=KNeighborsClassifier(n_neighbors=3)
print(knnmodel.fit(X_train,y_train))
y_predict1=knnmodel.predict(X_test)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_predict1)
print(acc)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test.values,y_predict1)
print(cm)
cm1=pd.DataFrame(data=cm,index=['setosa','versicolor','virginica'],columns=['setosa','versicolor','virginica'])
print(cm1)
prediction_output=pd.DataFrame(data=[y_test.values,y_predict1],index=['y_test','y_predict1'])
print(prediction_output.transpose())
prediction_output.iloc[0,:].value_counts()
Ks=50
mean_acc=np.zeros((Ks-1))
for n in range(1,Ks):
    neigh=KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1]=accuracy_score(y_test,yhat)
print(mean_acc)
print("The best accuracy was with",mean_acc.max(),"with k=",mean_acc.argmax()+1)
print(plt.plot(range(1,Ks),mean_acc,'g'))
print(plt.legend("Accuracy="))
print(plt.ylabel("Accuracy"))
print(plt.xlabel('Number of neighbors(K)'))
print(plt.tight_layout())
print(plt.show())