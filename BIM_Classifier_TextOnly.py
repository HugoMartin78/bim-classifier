# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:07:37 2015

@author: Hugo
"""

import numpy
import re
from scipy.sparse import hstack
from scipy import sparse


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC

# Chargement des données:
# Chaque ligne représente un objet de la maquette BIM
# Par colonne :
#   1.Nom de l'objet
#   2.Nom de son/ses matériaux
#   3.Nom de son/ses textures présent dans les matériaux
#   4.Nombre de polygon
#   5.Volume de l'objet
#   6.Label
#   7. Numéro de maquette
data = numpy.load('Data.npy')

# On traduit les labels string en valeur numérique
y = data[:,5]
y[y == 'S'] = 0
y[y == 'F'] = 1
y[y == 'I'] = 2
y[y == 'P'] = 3
y[y == 'M'] = 4


# récuperation des numéroes de modèles.
models = data[:,6].astype(numpy.float)

# data ne garde que la variable nom
data = numpy.delete(data,[1,2,3,4,5,6],1)

# Pour les trois première colonnes (TXT) on ne garde que les caractères standards.
for i in range(0,len(data)):
        data[i] = re.sub('[^AZERTYUIOPQSDFGHJKLMWXCVBNazertyuiopqsdfghjklmwxcvbn]', '', data[i]).lower()
        
# Delcaration des pre-processing / classifier
ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=0)
transformer = TfidfTransformer()

# Chaque paramètres représente la meilleur performance de l'ensemble
classifiers = [
    SGDClassifier(alpha = 1e-4),
    DecisionTreeClassifier(max_depth=None),
    SVC(gamma=2, C=1),
    RandomForestClassifier(n_estimators=60),
    AdaBoostClassifier()]
    

Result = numpy.empty((0,3), float)

# Boucle sur les classifiers
for clf in classifiers:

    Scores = numpy.array([])

    # Boucle One leave out
    for i in range(0,20):
        
        # On definie l'échatillon de test et d'apprentissage
        Sample = data[numpy.where(models != i)]
        L_Sample = y[numpy.where(models != i)]
        
        Test = data[numpy.where(models == i)]
        L_Test = y[numpy.where(models == i)]
 
        # Traitement du Text tri-gram + tfidf pour chaque variable textuelle
        ngram = ngram_vectorizer.fit_transform(Sample)
        T_ngram = ngram_vectorizer.transform(Test)
        
        X_train = transformer.fit_transform(ngram)
        X_test = transformer.transform(T_ngram)
        
        # On classe
        clf.fit(X_train, L_Sample.astype(numpy.int32))
        Scores = numpy.append(Scores,clf.score(X_test,L_Test.astype(numpy.int32)))
        
        
    Result  = numpy.append(Result,[numpy.mean(Scores),numpy.median(Scores),numpy.std(Scores)])


numpy.save('Result.npy',Result)

            
            
            
            
            
            
        
    
            
                
                