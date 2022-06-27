from configparser import Interpolation
import pandas as pd
import numpy as np
import pickle

data = pd.read_csv("thydata.csv")

data.Target = np.where(data.Target >=1,'1', '0')
print(data.isnull().sum())
# Replacing NaN value by mean, median depending upon distribution
data['TSH'].fillna(data['TSH'].mean(), inplace=True)
data['T3'].fillna(data['T3'].mean(), inplace=True)
data['TT4'].fillna(data['TT4'].median(), inplace=True)
data['T4U'].fillna(data['T4U'].median(), inplace=True)
data['FTI'].fillna(data['FTI'].median(), inplace=True)
# Model Building
from sklearn.model_selection import train_test_split
X = data.drop(columns='Target')
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'thyroid-detection-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))