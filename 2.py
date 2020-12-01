#1 - linear regression with sklearn

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#predictor variable
x = np.array([5, 15 , 25 , 35 , 45 ,55])
xm = x
x = x.reshape((-1,1))
x.shape

#response variable
y = np.array([4,7,23,5,39,23])
ym = y
#y = y.reshape((-1,1))
y.shape

plt.scatter(x,y)

model = LinearRegression()
model.fit(x,y)

r_sq = model.score(x,y)
print('The coefficient of determination is : ', r_sq)

# the model has the formula of b0 + b1 x

print('Intercept :',model.intercept_) #b0 is the intercept
print('Slope :',model.coef_) #b1 is the slope

x_predict = x
y_predict = model.predict(x_predict)
print('Predicted response : ', y_predict)
print('Actual value of y : ',y)

plt.scatter(x,y)
plt.plot(x_predict,y_predict,'y')
plt.xlabel('Independent variable, x')
plt.ylabel('Dependent variable, y')
plt.title('Simple Linear Regression plot')
ymean = np.average(y)
ybar = np.array([ymean,ymean,ymean,ymean,ymean,ymean])
plt.plot(x,ybar,'c')
plt.show()

#2 - linear regression without sklearn

xmean = np.average(xm)
ymean = np.average(ym)
xymean = np.average(np.multiply(xm,ym))
xmeansq = xmean*xmean
xsqbar = np.average(np.multiply(xm,xm))

b1 = ((xmean*ymean) - xymean)/(xmeansq-xsqbar)
b0 = ymean - (b1*xmean)

print('intercept ',b0)
print('slope ',b1)

y_pred = b0 + b1*xm
print('predicted values for y ',y_pred)

ssr = np.sum(np.square(y_pred - ymean))
print('SSR ', ssr)

sse = np.sum(np.square(ym - y_pred))
print('SSE ', sse)

ssto = np.sum(np.square(ym - ymean))
print('SSTO ', ssto)

rs = ssr/ssto
print('R squared ',rs)

x = 2*np.random.rand(100,1)
y = 4+3*x+(2*np.random.rand(100,1))
plt.scatter(x,y)

x = 2*np.random.rand(100,1)
y = 4-3*x+(2*np.random.rand(100,1)) # << adding some noise or distribution to the line
plt.scatter(x,y)
