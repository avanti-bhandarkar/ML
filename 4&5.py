import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data_logistic_1.csv')
df.shape
df.head()

y1 = df['admitted']
x1 = df['gmat']
x1 = np.array(x1)
x1 = x1.reshape((-1,1))
x1.shape

model = LinearRegression()
model.fit(x1,y1)

b0 = model.intercept_
b1 = model.coef_

yhat = b0+(b1*x1)
print('\n yhat 1 \n')
print(yhat)
yhat = yhat - np.mean(yhat)
print('\n yhat 2 \n')
print(yhat)
sig = 1/ (1 + np.exp(-yhat))
print('\n sig \n')
print(sig)
sh = sig.shape
output = np.zeros((sh[0],1))

for i in range (0,sh[0]):
  if sig[i]>0.5:
    output[i]=1
  else:
    output[i]=0
print('\n output \n')
print(output)

y1 = np.array(y1)
y2 = y1.reshape((1,sh[0]))
output = output.reshape((1,sh[0]))
print('output',output,'\n')
print('admitted',y2)

error = 0
for i in range(0,sh[0]):
  if output[0][i]!= y2[0[i]]:
    error = error +1
print('\n error \n')
print(error)

plt.scatter(yhat,y2)
plt.plot(yhat,sig)

df = pd.read_csv('data_logistic.csv')
df.shape
df.head()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sn

x = df[['gmat','gpa','work_experience']]
y = df['admitted']

x.shape
y.shape

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)
x_test.shape

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = metrics.accuracy_score(y_test,y_pred)
print('model accuracy:' ,accuracy)

confusion_matrix = pd.crosstab(y_test,y_pred,rownames =['Actual'],colnames=['Predicted'])
sn.heatmap(confusion_matrix,annot=True)

"""#EXP 5"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm

df = pd.read_csv('MLR_data.csv')
df.shape
df.head()

plt.figure(figsize = (10,10))
plt.subplot(2,2,1)
plt.scatter(df['Interest_Rate'],df['Stock_Index_Price'],color = 'red')
plt.title('Stock Index Price vs Interest Rate',fontsize = 14)
plt.xlabel('Interest Rate',fontsize = 14)
plt.ylabel('Stock Index Price',fontsize = 14)
plt.grid(True)

plt.figure(figsize = (10,10))
plt.subplot(2,2,2)
plt.scatter(df['Unemployment_Rate'],df['Stock_Index_Price'],color = 'blue')
plt.title('Stock Index Price vs Unemployment Rate',fontsize = 14)
plt.xlabel('Unemployment Rate',fontsize = 14)
plt.ylabel('Stock Index Price',fontsize = 14)
plt.grid(True)

plt.figure(figsize = (10,10))
plt.subplot(2,2,3)
plt.scatter(df['Year'],df['Stock_Index_Price'],color = 'green')
plt.title('Stock Index Price vs Year',fontsize = 14)
plt.xlabel('Year',fontsize = 14)
plt.ylabel('Stock Index Price',fontsize = 14)
plt.grid(True)


plt.figure(figsize = (10,10))
plt.subplot(2,2,4)
plt.scatter(df['Month'],df['Stock_Index_Price'],color = 'yellow')
plt.title('Stock Index Price vs Month',fontsize = 14)
plt.xlabel('Month',fontsize = 14)
plt.ylabel('Stock Index Price',fontsize = 14)
plt.grid(True)

plt.figure(figsize = (10,10))
plt.subplot(2,2,1)
plt.scatter(df['Interest_Rate'],df['Unemployment_Rate'],color = 'red')
plt.title('Interest Rate vs Unemployment Rate',fontsize = 14)
plt.xlabel('Interest Rate',fontsize = 14)
plt.ylabel('Unemployment Rate',fontsize = 14)
plt.grid(True)

plt.figure(figsize = (10,10))
plt.subplot(2,2,2)
plt.scatter(df['Interest_Rate'],df['Month'],color = 'blue')
plt.title('Interest Rate vs Month',fontsize = 14)
plt.xlabel('Interest Rate',fontsize = 14)
plt.ylabel('Month',fontsize = 14)
plt.grid(True)

plt.figure(figsize = (10,10))
plt.subplot(2,2,3)
plt.scatter(df['Unemployment_Rate'],df['Month'],color = 'green')
plt.title('Unemployment Rate vs Month',fontsize = 14)
plt.xlabel('Unemployment Rate',fontsize = 14)
plt.ylabel('Month',fontsize = 14)
plt.grid(True)

#exclude year from the statistic testing and use a simple linear reg model with 1 dependent and 1 independent variable
# independent variable = interest rate

X1 = df['Interest_Rate'] 
Y1 = df['Stock_Index_Price']
X1 = np.array(X1)
X1 = X1.reshape((-1,1))

#regression with sklearn

regr1 = linear_model.LinearRegression()
regr1.fit(X1,Y1)
print('Intercept : ', regr1.intercept_)
print('Slope : ', regr1.coef_)

#prediction with sklearn

New_Interest_Rate = 2.75
New_Interest_Rate = np.array(New_Interest_Rate)
New_Interest_Rate = New_Interest_Rate.reshape((-1,1))
print('Predicted stock index price : ', regr1.predict(New_Interest_Rate))

#with statsmodel

X1 = sm.add_constant(X1)
model1 = sm.OLS(Y1,X1).fit()
predictions = model1.predict(X1)

print_model1 = model1.summary()
print(print_model1)

#independent var = unemployment rate

X2 = df['Unemployment_Rate'] 
Y2 = df['Stock_Index_Price']
X2 = np.array(X2)
X2 = X2.reshape((-1,1))

#regression with sklearn

regr2 = linear_model.LinearRegression()
regr2.fit(X2,Y2)
print('Intercept : ', regr2.intercept_)
print('Slope : ', regr2.coef_)

#prediction with sklearn

New_Interest_Rate2 = 2.75
New_Interest_Rate2 = np.array(New_Interest_Rate2)
New_Interest_Rate2 = New_Interest_Rate2.reshape((-1,1))
print('Predicted stock index price : ', regr2.predict(New_Interest_Rate2))

#with statsmodel

X2 = sm.add_constant(X2)
model2 = sm.OLS(Y2,X2).fit()
predictions = model2.predict(X2)

print_model2 = model2.summary()
print(print_model2)

#independent var = month

X3 = df['Month'] 
Y3 = df['Stock_Index_Price']
X3 = np.array(X3)
X3 = X3.reshape((-1,1))

#regression with sklearn

regr3 = linear_model.LinearRegression()
regr3.fit(X3,Y3)
print('Intercept : ', regr3.intercept_)
print('Slope : ', regr3.coef_)

#prediction with sklearn

New_Interest_Rate3 = 2.75
New_Interest_Rate3 = np.array(New_Interest_Rate3)
New_Interest_Rate3 = New_Interest_Rate3.reshape((-1,1))
print('Predicted stock index price : ', regr3.predict(New_Interest_Rate3))

#with statsmodel

X3 = sm.add_constant(X3)
model3 = sm.OLS(Y3,X3).fit()
predictions = model3.predict(X3)

print_model3 = model3.summary()
print(print_model3)

X4 = df[['Interest_Rate','Unemployment_Rate']]
Y4 = df['Stock_Index_Price']

#regression with sklearn

regr4 = linear_model.LinearRegression()
regr4.fit(X4,Y4)
print('Intercept : ', regr4.intercept_)
print('Slope : ', regr4.coef_)

#prediction with sklearn

New_Interest_Rate4 = 2.75
New_Unemployment_Rate = 5.3

print('Predicted stock index price : ', regr4.predict([[New_Interest_Rate4,New_Unemployment_Rate]]))

#with statsmodel

X4 = sm.add_constant(X4)
model4 = sm.OLS(Y4,X4).fit()
predictions = model4.predict(X4)

print_model4 = model4.summary()
print(print_model4)

X5 = df[['Interest_Rate','Month']]
Y5 = df['Stock_Index_Price']

#regression with sklearn

regr5 = linear_model.LinearRegression()
regr5.fit(X5,Y5)
print('Intercept : ', regr5.intercept_)
print('Slope : ', regr5.coef_)

#prediction with sklearn

New_Interest_Rate5 = 3.5
New_Month = 4

print('Predicted stock index price : ', regr5.predict([[New_Interest_Rate5,New_Month]]))

#with statsmodel

X5 = sm.add_constant(X5)
model5 = sm.OLS(Y5,X5).fit()
predictions = model5.predict(X5)

print_model5 = model5.summary()
print(print_model5)

X6 = df[['Unemployment_Rate','Month']]
Y6 = df['Stock_Index_Price']

#regression with sklearn

regr6 = linear_model.LinearRegression()
regr6.fit(X6,Y6)
print('Intercept : ', regr6.intercept_)
print('Slope : ', regr6.coef_)

#prediction with sklearn

New_Unemployment_Rate6 = 5.0
New_Month1 = 2

print('Predicted stock index price : ', regr6.predict([[New_Unemployment_Rate6,New_Month1]]))

#with statsmodel

X6 = sm.add_constant(X6)
model6 = sm.OLS(Y6,X6).fit()
predictions = model6.predict(X6)

print_model6 = model6.summary()
print(print_model6)

X7 = df[['Unemployment_Rate','Month','Interest_Rate']]
Y7 = df['Stock_Index_Price']

#regression with sklearn

regr7 = linear_model.LinearRegression()
regr7.fit(X7,Y7)
print('Intercept : ', regr7.intercept_)
print('Slope : ', regr7.coef_)

#prediction with sklearn

New_Interest_Rate7 = 4.76
New_Unemployment_Rate7 = 2.4
New_Month2 = 9

print('Predicted stock index price : ', regr7.predict([[New_Interest_Rate7,New_Unemployment_Rate7,New_Month2]]))

#with statsmodel

X7 = sm.add_constant(X7)
model7 = sm.OLS(Y7,X7).fit()
predictions = model7.predict(X7)

print_model7 = model7.summary()
print(print_model7)
