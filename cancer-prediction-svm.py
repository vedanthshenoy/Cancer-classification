#importing all dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split     
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#reading data
k=pd.read_csv('cancer-data.csv')
data=pd.read_csv('cancer-data.csv',usecols=(2,5,6,7,8,11))
labels=k['diagnosis']

#splitting data into training and testing with training set 80% and test set 20%
x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2,random_state=4)

#Support Vector Classifier model
model=SVC()
model.fit(x_train,y_train)

#prediction
prediction_whole=model.predict(x_test)
prediction=model.predict([[13,519.8,0.1273,0.01932,0.1859,0.7389]])  
print(prediction)
#accuracy
print("Accuracy of this model is :"+str(accuracy_score(prediction_whole,y_test)*100)+" %")