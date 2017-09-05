from sklearn import tree
import numpy as np
from scipy.spatial import distance
import random

def dis(a,b):
    return distance.euclidean(a,b)

class RC():
    def fit(self,x_train,y_train):
        self.x_train=x_train
        self.y_train=y_train
    def predict(self,x_test):
        predictions=[]
        for i in x_test:
            label=random.choice(self.y_train)
            predictions.append(label)
        return predictions

class Knn():
    def fit(self,x_train,y_train):
        self.x_train=x_train
        self.y_train=y_train

    def closest(self,row):
        best_dist=dis(row,self.x_train[0])
        best_index=0
        for i in range(1,len(self.x_train)):
            dist=dis(row,x_train[i])
            if dist < best_dist:
                best_dist=dist
                best_index=i
        return self.y_train[best_index]

    def predict(self,x_test):
        predictions=[]
        for i in x_test:
            label = self.closest(i)
            predictions.append(label)
        return predictions



from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
#training data
#train_target=np.delete(iris.target,test_index)
#train_data=np.delete(iris.data,test_index,axis=0)

#testing data
#test_target=iris.target[test_index]
#test_data=iris.data[test_index]

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)

classifier=Knn()
classifier.fit(x_train,y_train)

prediction = classifier.predict(x_test)
print(prediction)

from sklearn.metrics import accuracy_score
print (accuracy_score(y_test,prediction))

