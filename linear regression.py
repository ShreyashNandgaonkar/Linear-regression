#Linear regression using sklearn
from sklearn import linear_model
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
#load dataset

d= datasets.load_boston()
#dataexploration
d
#import packages
import numpy as np
import pandas as pd
#define predictors and load dataset in pandas dataframe
df = pd.DataFrame(d.data,columns=d.feature_names)
df.head()
df.info()
df.describe()
sns.pairplot(df)
#putting target variable(MEDV) in another dataframe
t = pd.DataFrame(d.target,columns=["MEDV"])
# predictor(dependant) is df and Target(independant) is t
#define X and y for our model
X= df
y=t["MEDV"]
#Model
lm=linear_model.LinearRegression()
#fitting model
model=lm.fit(X,y)
#prediction
prediction=lm.predict(X)
print(prediction)
#score
lm.score(X,y)
#coefficients
lm.coef_
#intercept
lm.intercept_
