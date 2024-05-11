import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
df=sns.load_dataset("mpg")
df
df.isnull().sum()
df.dropna().sum()
df.dropna(inplace=True)
df.isnull().sum()
x=df[['cylinders',	'displacement',	'horsepower',	'weight']]
y=df.mpg
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
filename= 'mpg_regression.sav'
pickle.dump(model, open(filename, 'wb'))