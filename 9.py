
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score,precision_recall_curve
from sklearn.preprocessing import StandardScaler
import seaborn as sn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict

data=pd.read_csv('Naive_Bayes.csv')
data.shape
data.head()

#Convert categorial variable to numeric
data['Sex_cleaned']=np.where(data['Sex']=='male',0,1)
data.head()
data['Embarked_cleaned']=np.where(data['Embarked']=='S',0,np.where(data['Embarked']=='C',1,np.where(data['Embarked']=='Q',2,3)))
data.head()

#handling missing values
data=data[['Survived','Pclass','Sex_cleaned','Age','SibSp','Parch','Fare','Embarked_cleaned']].dropna(axis=0,how='any')
data.shape
data.head()

#selecting independent and dependent variables
X = data[['Pclass','Sex_cleaned','Age','SibSp','Parch','Fare','Embarked_cleaned']]
y = data['Survived']

print(X.shape)
print(y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 10)

print(X_train.shape)
print(X_test.shape)

#featurescaling

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_train

from sklearn import metrics
scores = {}
scores_list = []
for k in range(1,30):
  classifier1 = KNeighborsClassifier(n_neighbors=k)
  classifier1.fit(X_train,y_train)
  y_pred1 = classifier1.predict(X_test)
  temp = metrics.accuracy_score(y_test,y_pred1)
  scores[k] = temp
  scores_list.append(temp)

k_range = range(1,30)
plt.plot(k_range,scores_list)
plt.title('accuracy of classifier for various values of k')
plt.xlabel('value of k')
plt.ylabel('accuracy')

classifier = KNeighborsClassifier(n_neighbors = 12)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

confusion_matrix = pd.crosstab(y_test,y_pred, rownames = ['actual'],colnames = ['predicted'])
sn.heatmap(confusion_matrix, annot = True)

print('Precision:',precision_score(y_test,y_pred))

print('Recall:',recall_score(y_test,y_pred))

print('F1 score:',f1_score(y_test,y_pred))

#Receiver operating chracteristics / ROC Curve
y_scores = cross_val_predict(classifier,X_train,y_train,cv=3,method ='predict')
precision, recall, thresholds = precision_recall_curve(y_train,y_scores)
