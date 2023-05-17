
import warnings
warnings.filterwarnings('ignore')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data = pd.read_csv('Salary_Data.csv')
data.head(10)



data.shape




data.describe()

sns.pairplot(y_vars = 'Salary', x_vars = 'YearsExperience' , data = data)



data.corr()


x = data['YearsExperience']
y = data['Salary']





x_train, x_test, y_test, y_train = train_test_split(x,y,train_size = 0.7, test_size = 0.3, random_state = 100)




x_train.shape




x_test.shape

