#!/usr/bin/python

import pandas as pd 
import numpy as np 
from pandas import Series, DataFrame
import tensorflow as tf

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

# use random forest to fill some of the missing data
from sklearn.ensemble import RandomForestRegressor 
def set_missing_ages(df):
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    
    y = known_age[:, 0]
    X = known_age[:, 1:]
 
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    predictedAges = rfr.predict(unknown_age[:, 1::])
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges  
    return df, rfr

# deal with features with half of missing value
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df


# handling the training data
data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
#data_test = set_Cabin_type(data_test)
#data_test = set_missing_ages(data_test)

# set one-hot encode for no-nonnumeric features
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

#datasets re-construction to the copies of df
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# features scaler
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()

#reshape features before put into scaler
age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), age_scale_param)

fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)

train_df = df.filter(regex="Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*")
train_np = train_df.values


# handling the test data
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

age_scale_param_test = scaler.fit(df_test['Age'].values.reshape(-1,1))
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param_test)
fare_scale_param_test = scaler.fit(df_test['Fare'].values.reshape(-1,1))
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param_test)
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_np = test.values
x_test = test_np[:, 0:]




x_train = train_np[:, 1:]
y_train = train_np[:, 0]


from sklearn import linear_model
#clf.fit(x_train, y_train)

from sklearn.ensemble import BaggingRegressor
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(x_train, y_train)

predictions = bagging_clf.predict(x_test)
predictions = predictions.astype(np.int32)
data_test['Survived'] = predictions
data_test.drop(['Pclass'], axis=1, inplace=True)
data_test.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
data_test.drop(['Age', 'SibSp', 'Parch', 'Fare'], axis=1, inplace=True)

data_test.to_csv("result.csv",index=False,sep=',')  
