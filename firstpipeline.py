print("in this doc should be implemented my first pipeline")
print("\nDas Beispiel von towards Data Sience\n")

#Einlesen des Datensatzes

import pandas as pd
filepath = 'winequality-red.csv'
winedf = pd.read_csv(filepath, sep=';')

print(winedf.isnull().sum())
#alle Merkmale werden einmal dargestelt

print(winedf.head(5))
#die ersten 5 Zeilen werden dargestellt

#Trennen der Eigenschaften und Labels vom Datenset

X = winedf.drop(['quality'], axis=1)
#Ausschneiden der letzten Spalte 'quality'
#print('-----------X')
#print(X)

Y = winedf['quality']
#speichern der Werte aus der letzten zuvor gelöschten Spalte
#print('-----------Y')
#print(Y)

from sklearn.svm import SVC
#C-Support Vector Classification ist ein estimator
from sklearn.preprocessing import StandardScaler
#StandardScaler ist ein Transformer

steps = [('Scaler', StandardScaler()), ('SVM', SVC())]
#Definition des zu verwendenden Estimator und Transformer

from sklearn.pipeline import Pipeline
pipeline = Pipeline(steps) #definition des Pipeline Objekts.


#Aufteilung des Datensets in Training und Testdaten
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=30, stratify=Y)

print('Das sind die Verteilungen der labels')
print(winedf['quality'].value_counts())

parameters= {'SVM__C':[0.001, 0.1, 10, 100, 10e5], 'SVM__gamma':[0.1, 0.01]}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
grid.fit(X_train, y_train)

print("Score = %3.2f" %(grid.score(X_test, y_test)))
print(grid.best_params_)