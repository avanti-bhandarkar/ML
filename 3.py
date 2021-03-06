import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from google.colab import files
uploaded = files.upload()

data = pd.read_csv('sat_cgpa.csv')

data.describe()
data.head()

y = data['GPA']
x1 = data['SAT']

plt.scatter(x1,y)
plt.title('Scatter plot of GPA and SAT scores')
plt.xlabel('SAT')
plt.ylabel('GPA')

x = sm.add_constant(x1)

print(x)

results = sm.OLS(y,x).fit()
#OLS = ordinary least squares method os regression
#OLS fn takes response variable y first and then the predictor variable x

results.summary()

yhat = 0.2750 + 0.0017*x1
plt.scatter(x1,y)
plt.plot(x1,yhat,lw=2,c='r')

data.describe()

# for intercept

print('For intercept \n')
C_up = 3.330238 + (0.673*0.271)/ np.sqrt(84)
print('Upper limit of confidence interval is : ',C_up)

C_down = 3.330238 - (0.673*0.271)/ np.sqrt(84)
print('Lower limit of confidence interval is : ',C_down)

print('\n')

#for slope

print('For slope \n')
C_up = 0.0017 + (7.487*0.000)
print('Upper limit of confidence interval is : ',C_up)

C_down = 0.0017 - (7.487*0.000)
print('Lower limit of confidence interval is : ',C_down)
