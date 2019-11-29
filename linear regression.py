#Linear regression using sklearn
from sklearn import linear_model
from sklearn import datasets
#load dataset
d= datasets.load_boston()
#check the data
d
#import packages
import numpy as np
import pandas as pd
#define predictors and load dataset in pandas dataframe
df = pd.DataFrame(d.data,columns=d.feature_names)
#putting target variable(MEDV) in another dataframe
t = pd.DataFrame(d.target,columns=["MEDV"])
# predictor(dependant) is df and Target(independant) is t
#fdefine X and y for our model
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