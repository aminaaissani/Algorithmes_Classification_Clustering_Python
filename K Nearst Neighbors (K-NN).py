#Importation des Librairies 
import pandas as pd
import numpy as np

#chargement des données
iris=pd.read_csv('iris.csv')
print(iris.head())

#Création d'un tableau de donnée X et un tableau des classe Y
feature_columns = ['sepal.length', 'sepal.width', 'petal.length','petal.width']
x = iris[feature_columns].values
y = iris['variety'].values

#Convertir les labels de texte à des nombres
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(set(y.tolist()))

#Normalisation des données, pour éviter que les valeurs abérrantes influencent le résultat
from sklearn import preprocessing
x = preprocessing.scale(x)

#Vérification des données
iris.info()

print("\nLa taille et la dimension des données", np.size(x), np.shape(x))

print("\nLa taille et la dimension des classe", np.size(y), np.shape(y))

print("\nLes informations\n",iris.info)

print("\nStatistiques descriptives\n",iris.describe())

#Appliquer le calssificateur KNN sur les données iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#importation de metrics - utilisé pour les mesures de performances
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import label_binarize
#Utilisez label_binarize pour être indiquer que les classes sont de type multi-étiquettes
y = label_binarize(y, classes=[0, 1, 2])

score=list()
#subdivision des données – 75% pour l'entrainement et 25% pour le test, on mit shuffle vrai pour mélanger les données 
x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=0.25 ,shuffle='true')


#on pose k=3 car on a 3 classes de fleurs
knn= KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

score=round(accuracy_score(y_test,y_pred),4)
print('Taux de bonne classification',score)

#Application de la méthode de validation croisée 10-fold 
from sklearn.model_selection import cross_val_score

#évaluation en validation croisée : 10 cross-validation
succes = cross_val_score(knn,x,y,cv=10,scoring='accuracy')
print(succes)

#Calcul précision, rappel et f-mesure
from sklearn.metrics import recall_score,average_precision_score,f1_score

#La macro-moyenne calcule d'abord la précision et le rappel sur chaque classe i suivie d'un calcul de
#la moyenne des précisions et des rappels sur les n classes.

#Sensibilité (ou rappel) est la proportion des items pertinents parmi l'ensemble des items pertinents.
rappel = recall_score(y_test,y_pred,average='macro')
print('Rappel est: ',rappel)

#précision (ou valeur prédictive positive) est la proportion des items pertinents parmi l'ensemble des items proposés
#précision
average_precision = average_precision_score(y_test, y_pred,average='macro')
print('Précision est: ',average_precision)

#f-mesure: Une mesure qui combine la précision et le rappel est leur moyenne harmonique, nommée F-mesure ou F-score 
#f-mesure=2* (précision*rappel)/(précision+rappel)
f1_score=f1_score(y_test, y_pred,average='macro')
print('F-mesure est: ',f1_score)

#Calcul matrice de confusion
from sklearn.metrics import confusion_matrix

matrice_conf=confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(matrice_conf)
