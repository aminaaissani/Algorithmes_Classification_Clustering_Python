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

"""2- Description et analyse de la base téléchargée
- Vérifier s'il y a une incohérence dans l'ensemble de données comme, il n'y a pas de valeurs nulles dans l'ensemble de données, donc les données peuvent être traitées
"""

iris.info()

print("\nLa taille et la dimension des données", np.size(x), np.shape(x))

print("\nLa taille et la dimension des classe", np.size(y), np.shape(y))

print("\nLes informations\n",iris.info)

print("\nStatistiques descriptives\n",iris.describe())

"""3- Description de l'approche des k-plus proches voisins

Il s’agit d’un algorithme d’apprentissage supervisé. Il sert aussi bien pour la classification que la régression.

4- Appliquer le calssificateur KNN sur les données iris

---
"""

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

#choix du k optimale
score = []
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    score.append(round(accuracy_score(y_test,y_pred),4))
print(score)
# Graphe
plt.plot(range(1, 30), score)
plt.title('Nombre K')
plt.xlabel('K')
plt.ylabel('score') 
plt.show()

"""Le graphe peut changer car il dépends de la division des données proposée par train_test_split.

Pour cela on le k est calculé dynamiquement.
"""

score_max=0
k=0
for i in range(len(score)):
  if score_max < score[i]:
     k=i+1
     score_max=score[i]

print('le k optimale est ',k, ' avec précision est:',score_max)

knn= KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)

score=round(accuracy_score(y_test,y_pred),4)
print('Taux de bonne classification',score)

"""5- Application de la méthode de validation croisée (10-fold)"""

#Application de la méthode de validation croisée 10-fold 
from sklearn.model_selection import cross_val_score

#évaluation en validation croisée : 10 cross-validation
succes = cross_val_score(knn,x,y,cv=10,scoring='accuracy')
print(succes)

"""Calcul du rappel et de la précision"""

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

"""6- Matrice de Confusion"""

from sklearn.metrics import confusion_matrix

matrice_conf=confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(matrice_conf)

"""7- Analyse des résultats obtenus

On analysant les résultats (score, rappel....) on peut conclure que KNN a donner des bons résultats lorsque la taille des données n'est pas très grande.

## TP2

1- Importation des données & Chargement des fichiers

URLs: 

X_train:https://drive.google.com/open?id=1tD8dbG7NyfpEUjTYE1u5eQbivJ_89K5A

Y_train:https://drive.google.com/open?id=1JuRjgpsHBvfIAYIZpT3gHdmkDPUB4tb2

X_test: https://drive.google.com/open?id=1ccMol-khJ__ETJqyXgXEullPBzEbB6Wp

Y_test:https://drive.google.com/open?id=1ABur1oFf43o3c7p7L348b5IyDV-GYw3q
"""

import pandas as pd
import numpy as np

#Importation des données d'entrainement
X_train=pd.read_fwf('X_train.txt')
Y_train=pd.read_fwf('y_train.txt')

#Importation des données de test
X_test=pd.read_fwf('X_test.txt')
Y_test=pd.read_fwf('y_test.txt')

print(X_train.head())
print(Y_train.head())

"""2- Description de la base de données"""

#Nombre et noms de colonnes
print(X_train.columns)
print(Y_train.columns)

"""Description X_train"""

print('\n Le format des données',np.shape(X_train))

print('\n Les statiqtiques : \n\n', X_train.describe() )

"""Description de Y_train"""

print('\n Le format des données',np.shape(Y_train))
print("\n Le nombre par classe \n",Y_train.groupby('5').size())

"""3- Application de troix approches de classification supervisée

Le choix est porté sur les modèles suivants: KNN, SVM, Random Forest

1- Application du modèle KNN sur les données
"""

#importation des libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

"""- Choix du k optimale"""

score_k = []
for i in range(1, 29):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,np.ravel(Y_train))
    Y_pred=knn.predict(X_test)
    score_k.append(round(accuracy_score(Y_test,np.ravel(Y_pred)),4))

# Graphe
plt.plot(range(1, 29), score_k)
plt.title('Nombre K')
plt.xlabel('K')
plt.ylabel('score') 
plt.show()

score_max=0
k=0
for i in range(len(score_k)):
  if score_max < score_k[i]:
     k=i+1
     score_max=score_k[i]

print('le k optimale est ',k, ' avec précision est:',score_max)

"""La meilleure précision est donnée lorsque la valeur de k est égale à 8. Pour la suite nous fixons k=8."""

knn=KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,np.ravel(Y_train))
Y_pred_knn=knn.predict(X_test)

score= round(accuracy_score(Y_test,np.ravel(Y_pred_knn)),4)
print('La Précision du modèle KNN est:',score)

"""2-  SVM"""

svm = svm.SVC(gamma='auto')
svm.fit(X_train, np.ravel(Y_train))
Y_pred_svm=svm.predict(X_test)
score= round(accuracy_score(Y_test,np.ravel(Y_pred_svm)),4)
print('La Précision du modèle SVM est:',score)

"""3- Random Forest"""

rforest = RandomForestClassifier(n_estimators=10)
rforest.fit(X_train, np.ravel(Y_train))
Y_pred_rforest=rforest.predict(X_test)
score=round(accuracy_score(Y_test,np.ravel(Y_pred_rforest)),4)
print('La Précision du modèle Radom Forest est:',score)

"""4- Validation croisée

On utilise la base d'appretissage X_train et Y_train pour appliquer la validation croisée.
"""

#Application de la méthode de validation croisée 10-fold 
from sklearn.model_selection import cross_val_score

#Evaluation en validation croisée : 10 cross-validation
Knn_Score = cross_val_score(knn,X_train,np.ravel(Y_train),cv=10,scoring='accuracy')
print(Knn_Score)

sommeScore=0
for i in range(len(Knn_Score)):
  sommeScore=sommeScore+Knn_Score[i]
print('la moyenne des scores est', round(sommeScore/len(Knn_Score),4))

svm_score = cross_val_score(svm,X_train,np.ravel(Y_train),cv=10,scoring='accuracy')
print(svm_score)
sommeScore=0
for i in range(len(svm_score)):
  sommeScore=sommeScore+svm_score[i]
print('la moyenne des scores est', round(sommeScore/len(svm_score),4))

rforest_score = cross_val_score(rforest,X_train,np.ravel(Y_train),cv=10,scoring='accuracy')
print(rforest_score)
sommeScore=0
for i in range(len(rforest_score)):
  sommeScore=sommeScore+rforest_score[i]
print('la moyenne des scores est', round(sommeScore/len(rforest_score),4))

"""Calcul Rappel,Précision et F-mesure"""

from sklearn.metrics import recall_score,average_precision_score,f1_score
from sklearn.preprocessing import label_binarize

#KNN
rappel_knn = recall_score(Y_test,Y_pred_knn,average='macro')
#SVM
rappel_svm = recall_score(Y_test,Y_pred_svm,average='macro')
#RandomForest
rappel_rforest= recall_score(Y_test,Y_pred_rforest,average='macro')
rappel=[rappel_knn, rappel_svm,rappel_rforest]
print(rappel)

Y_Test= label_binarize(Y_test, classes=[1,2,3,4,5,6])
Y_knn= label_binarize(Y_pred_knn, classes=[1,2,3,4,5,6])
Y_svm= label_binarize(Y_pred_svm, classes=[1,2,3,4,5,6])
Y_rforest= label_binarize(Y_pred_rforest, classes=[1,2,3,4,5,6])


precision_knn = average_precision_score(Y_Test, Y_knn,average='macro')
precision_svm = average_precision_score(Y_Test, Y_svm,average='macro')
precision_rforest = average_precision_score(Y_Test, Y_rforest,average='macro')
precision=[precision_knn,precision_svm,precision_rforest]
print(precision)

f1_score_knn=f1_score(Y_test, Y_pred_knn,average='macro')
f1_score_svm=f1_score(Y_test, Y_pred_svm,average='macro')
f1_score_rforest=f1_score(Y_test, Y_pred_rforest,average='macro')
f1_score=[f1_score_knn,f1_score_svm,f1_score_rforest]
print(f1_score)

"""Visualisation des résultats"""

results=[rappel,precision,f1_score]
results_df = pd.DataFrame(results,
                                index = ['Rappel', 'Précision', 'F-mesure'],
                                columns = ['knn', 'svm', 'Random_forest'])
print(results_df)

"""5-Matrice de confusion"""

from sklearn.metrics import confusion_matrix


matrice_conf_knn=confusion_matrix(Y_test, np.ravel(Y_pred_knn))
print('Matrice de confusion de knn:\n')
print(matrice_conf_knn)

matrice_conf_svm=confusion_matrix(Y_test, np.ravel(Y_pred_svm))
print('\nMatrice de confusion de svm:\n')
print(matrice_conf_svm)

matrice_conf_rforest=confusion_matrix(Y_test, np.ravel(Y_pred_rforest))
print('\nMatrice de confusion de random forest:\n')
print(matrice_conf_rforest)

"""6-Comparaison des classificateurs

En analysant la précision de chaque modèle, le tableau de comparaison (rappel, précision, f-mesure) ainsi les matrices des confusions, nous pouvons dire que le modèle qui donne la meilleure classification est SVM suivi par KNN et enfin le Random Forest.

7- Proposition d'une méthodologie pour des meilleurs résultats

On utilise le Voting classifier pour obtenir le meilleur des résultats.
"""

from sklearn.ensemble import VotingClassifier

clf = VotingClassifier([('lsvc',svm.SVC(gamma='auto')),
                            ('knn',KNeighborsClassifier(n_neighbors=k)),
                            ('rfor',RandomForestClassifier(n_estimators=10))],voting='hard')
clf.fit(X_train, np.ravel(Y_train))
score = clf.score(X_test, np.ravel(Y_test))
print('La Précision avec Voting Classifier est:',round(score,6))
