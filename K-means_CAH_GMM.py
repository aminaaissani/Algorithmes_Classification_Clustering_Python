!wget https://drive.google.com/open?id=1HLHm9WRZvAOkEERotIDr3tjloeAWrPdt -O Iris.csv

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



#chargement des données
df=pd.read_csv('Iris.csv',error_bad_lines=False)
print(type(df))

"""Affichage des données, pour mieux comprendre le jeu de données."""

print("Afficher le debut de la liste\n",df.head())

print("\nLes informations\n",df.info)

print("\nDimension des données\n",df.shape)

print("\nStatistiques descriptives\n",df.describe())

print("\nLes groupes \n",df.groupby('Species').size())

print("\nLes columns\n", df.columns)

print("\n \n", df["Species"].value_counts())

#Il n'y a d'incohérence dans l'ensemble de données et pas de valeurs nulles, donc les données peuvent être traitées

df.info()


#Fractionner les données en deux tableaux X :  fonctionnalités et y : étiquettes.

feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = df[feature_columns].values
y = df['Species'].values

# Deuxiéme Methode en Utilisant iloc
# x = df.iloc[:,[1,2,3,4]].values
# y = df.iloc[:,-1]

"""Transformer la sortie Species en numérique :
  0 : Iris setosa,
  1 : Iris versicolor,
  2 : Iris virginica.
"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(set(y.tolist()))

"""Centrage réduction des données:

pour éviter que variables à forte variance pèsent indûment sur les résultats
"""

from sklearn import preprocessing
X = preprocessing.scale(X)

"""Présentation graphique des données"""

plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel(y[0])
plt.ylabel(y[1])

plt.tight_layout()
plt.show()

"""Pour le choix d'un nombre optimal de clusters,On s'est basé sur deux méthode du calcul :
intertie et la silhouette.

Premiere Methode : inertie.
"""

from sklearn.cluster import KMeans
c = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300 ,n_init=10, random_state=0)
    kmeans.fit(X)
    c.append(kmeans.inertia_)

# Graphe
plt.plot(range(1, 11), c)
plt.title('Nombre K')
plt.xlabel('K')
plt.ylabel('C') 
plt.show()

"""Deuxieme Methode : silhouette.
Le tracé de silhouette affiche une mesure de la proximité de chaque point d'un cluster avec les points des clusters voisins et fournit ainsi un moyen d'évaluer visuellement des paramètres tels que le nombre de clusters. Cette mesure a une plage de [-1, 1].
"""

from sklearn import metrics

#utilisation de la métrique "silhouette" pour faire varier le nombre de clusters de 2 à 10
d = np.arange(11,dtype="double")
for k in np.arange(11):
 km = KMeans(n_clusters=k+2)
 km.fit(X)
 d[k] = metrics.silhouette_score(X,km.labels_)
print(d)

#graphique
#justification du choix de la silhouette
plt.title("Silhouette")
plt.xlabel("nb of clusters")
plt.plot(np.arange(2,13,1),d)
plt.show()

"""Selon les deux methodes le nombre optimal de clusters est le point représentant le coude. Ici le coude peut être représenté par K valant 3 ou 4.
Nous prenons le nombre opimale de clusters = 3 et nous passons a l'application du modele.
"""

#On spécifie le paramètre init=k-means++, qui sélectionne les centres de cluster initiaux pour le clustering k-mean de 
#manière intelligente pour accélérer la convergence. 
#n_init=10 par défaut  :Nombre de fois que l'algorithme k-means sera exécuté avec différentes graines de centroïde. 
#max_iter =300 par défaut: Nombre maximal d'itérations de l'algorithme k-means pour une seule exécution.

km = KMeans(n_clusters= 3, max_iter= 100, n_init=10, random_state=0)
ypred = km.fit_predict(X)
ypred

import seaborn as sns
score1 = metrics.silhouette_score(X, ypred)
print(score1)

from sklearn.metrics.cluster import adjusted_rand_score

score2 = adjusted_rand_score(y, ypred)
print(score2)

"""##Implémentation de CAH

Définition:

Importer les librairies necessaires
"""

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch

"""Générer la matrice des distance"""

#générer la matrice des liens
a = linkage(X,method='ward',metric='euclidean')

"""Affichage du dendrogramme."""

#affichage du dendrogramme
plt.title("CAH")
dendrogram(a,labels=df.index,orientation='left' ,color_threshold=12)
plt.show()

"""Découpage à t = 12 identifiants de 3 groupes"""

groupes_cah = sch.fcluster(a,t=12,criterion='distance') 
print(np.unique(groupes_cah).size, "groupes constitués")

score3 = adjusted_rand_score(y, groupes_cah)
print(score3)

"""Comparaison entre Kmeans et CAH."""

pd.crosstab(groupes_cah,kmeans.labels_)

"""##Implémentation de GMM

Définitions :

Importation des librairies.
"""

from sklearn.mixture import GaussianMixture

"""Application de modeles"""

gmm = GaussianMixture(n_components=3)
gmm.fit(X)

ypred2 = gmm.predict(X)
ypred2

"""Evaluation"""

from sklearn.metrics.cluster import adjusted_rand_score

score = adjusted_rand_score(y, ypred2)
score

"""**Conclusion **
On comparons et on analysons les résultats des scores donnés par le métrique "adjusted rand score" on conclue que l'algorithme GMM donne des meilleurs résultats que le modèle k-means et CAH.
"""
