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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

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

# data ne garde que les variables
data = numpy.delete(data,[5,6],1)

# Pour les trois première colonnes (TXT) on ne garde que les caractères standards.
for i in range(0,len(data)):
    for j in [0,1,2]:
        data[i,j] = re.sub('[^AZERTYUIOPQSDFGHJKLMWXCVBNazertyuiopqsdfghjklmwxcvbn]', '', data[i,j]).lower()
        
# Delcaration des pre-processing / classifier
ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=0)
transformer = TfidfTransformer()

# Chaque paramètres représente la meilleur performance de l'ensemble
classifiers = [
    SGDClassifier(alpha = 1e-4),
    DecisionTreeClassifier(max_depth=None),
    SVC(gamma=2, C=1),
    RandomForestClassifier(n_estimators=60),
    GaussianNB(),
    LDA(),
    QDA()]
    

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
        
        # On recupere en premier les variables numérique
        Num_weight = 0.03
        Num_train = Num_weight * sparse.csr_matrix(numpy.delete(Sample,[0,1,2],1).astype(numpy.float))
        Num_test = Num_weight * sparse.csr_matrix(numpy.delete(Test,[0,1,2],1).astype(numpy.float))
    
        tfidf = numpy.array([Num_train])
        T_tfidf = numpy.array([Num_test])       
        
        # Traitement du Text tri-gram + tfidf pour chaque variable textuelle
        for j in[0,1,2]:
            ngram = ngram_vectorizer.fit_transform(Sample[:,j])
            T_ngram = ngram_vectorizer.transform(Test[:,j])
            
            tfidf = numpy.append(tfidf,transformer.fit_transform(ngram)) 
            T_tfidf = numpy.append(T_tfidf,transformer.transform(T_ngram))
        
        # On stack toutes les variables entre elles
        X_train = hstack(tfidf)
        X_test = hstack(T_tfidf)
        
        # On classe
        clf.fit(X_train, L_Sample.astype(numpy.int32))
        Scores = numpy.append(Scores,clf.score(X_test,L_Test.astype(numpy.int32)))
        
        
    Result  = numpy.append(Result,[numpy.mean(Scores),numpy.median(Scores),numpy.std(Scores)])


numpy.save('Result.npy',Result)

            
            
            
            
            
            
        
    
            
                
                