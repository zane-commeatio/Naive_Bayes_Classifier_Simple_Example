'''
Number of times pregnant.
Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
Diastolic blood pressure (mm Hg).
Triceps skinfold thickness (mm).
2-Hour serum insulin (mu U/ml).
Body mass index (weight in kg/(height in m)^2).
Diabetes pedigree function.
Age (years).
Class variable (0 or 1).
'''

# TEST TEST

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
# Ignorer les warnings
import warnings
warnings.filterwarnings("ignore")

#Importation des données
data = pd.read_csv('pima.csv')
data.head()
data.columns = ['pregnant','glucose','blood_pressure','triceps_skinfold','insulin','mass','pedigree','age','label']


# Séparation features/target
X = data.iloc[:,0:8]
Y = data.iloc[:,-1]

X_scaled = X

# Verification de la distribution normale
sns.distplot(X[['pregnant']])

# Essayer de transformer en distribution normale
qt = QuantileTransformer(output_distribution='normal')
X_scaled[['pregnant']] = qt.fit_transform(X[['pregnant']])

# Ca change rien
sns.distplot(X_scaled[['pregnant']])

# Mise à l'echelle 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 


X_train, X_test, y_train ,y_test = train_test_split(X,Y,test_size=0.20)

# Modèle GaussianNB
clf = GaussianNB()
clf.fit(X_train,y_train)

y_pred = pd.DataFrame(clf.predict(X_test))

clf.score(y_pred,y_test)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_pred,y_test))

# COmparaison du GaussianNB avec RandomForest et LogisticRegression
def compare_models(data):
    
    data.columns = ['pregnant','glucose','blood_pressure','triceps_skinfold','insulin','mass','pedigree','age','label']

    X = data.iloc[:,0:8]
    Y = data.iloc[:,-1]    
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X = pd.DataFrame(X,columns=['pregnant','glucose','blood_pressure','triceps_skinfold','insulin','mass','pedigree','age'])
    scores = pd.DataFrame(columns=['GaussianNB','Random_Forest','Log_reg'])
    
    # 100 modélisations avec random seed split aléatoire
    for i in range(0,101):
    
        X_train, X_test, y_train ,y_test = train_test_split(X,Y,test_size=0.20)
        
        reg = linear_model.LogisticRegression()
        reg.fit(X_train,y_train)
        y_pred = pd.DataFrame(reg.predict(X_test))
        reg_score = reg.score(X_test,y_test) 
        
        clf = GaussianNB()
        clf.fit(X_train,y_train)
        y_pred = pd.DataFrame(clf.predict(X_test))
        gaussian_score = clf.score(y_pred,y_test)
        
        clf2 = RandomForestClassifier()
        clf2.fit(X_train,y_train)
        forest_score = clf2.score(X_test,y_test)
        
        
        # Ligne contenant les scores des 3 modèles à ajouter au DF de resultats
        line = { 'GaussianNB':gaussian_score, 'Random_Forest':forest_score, 'Log_reg':reg_score}
        # Ajout de la ligne au DF
        scores = scores.append(line,ignore_index=True)
    
    # Calcul des moyennes de scores pour chaque modèle
    gaussian_mean = scores.loc[:,'GaussianNB'].mean()
    forest_mean = scores.loc[:,'Random_Forest'].mean()
    logistic_mean = scores.loc[:,'Log_reg'].mean()
    
    
    print('\n \nScore moyen du Naive Bayes Network : ',gaussian_mean,'\nScore moyen du Random Forest : ',forest_mean,'\nScore moyen de la regression logistique : ',logistic_mean,'\n')
        

compare_models(data)

#%%




#%%
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix 

# Importation des données
df = pd.read_csv('SMSSpam',sep='\t',names=['Status','Message'])
df.head()
len(df[df.Status=='spam'])

# Transformer la targer en 0 et 1
df.loc[df.Status=='spam','Status'] = 0
df.loc[df.Status=='ham','Status'] = 1

# Séparation features/target
x = df.loc[0:1,'Message']
x = df.loc[:,'Message']


print(x)
text = x.loc[1]
print(text)
x.loc[1] = 'Ok world... Joking wif world...'

print(x)

# Vectorisation du texte avec la méthode Count
cv = CountVectorizer()
X_counted = cv.fit_transform(x)

# Visualisation du résultat de la vectorisation
print(X_counted)
X_counted.shape
print(type(X_counted))

feature_names = cv.get_feature_names()
print(feature_names)

# Mise en DF du resultat de vectorisation
counter_df = pd.DataFrame(X_counted.toarray(), columns=cv.get_feature_names())
print(counter_df)

#%%
# Autre vectorisation avec la méthode TF-IDF
vect = TfidfVectorizer()
X_tfid = vect.fit_transform(x)

# Visualisation des résultat de cette autre méthode
print(X_tfid)
X_tfid.shape
print(type(X_tfid))

feature_names = vect.get_feature_names()
print(feature_names)


tfid_df = pd.DataFrame(X_tfid.toarray(), columns=vect.get_feature_names())
print(tfid_df)

#%%

#%%

# 100 modélisations pour chaque méthode de vectoisation

df = pd.read_csv('SMSSpam',sep='\t',names=['Status','Message'])

df.loc[df.Status=='spam','Status'] = 0
df.loc[df.Status=='ham','Status'] = 1

# Création d'un DF vide pour contenir les résultats
df_results = pd.DataFrame(columns=['Count','TFIDF'])

for i in range(0,101):
    x = df.loc[:,'Message']
    Y = df.loc[:,'Status']
    
    cv = CountVectorizer()
    X = cv.fit_transform(x)
    
    X_train, X_test, y_train ,y_test = train_test_split(X,Y,test_size=0.20)
    
    #print(y_train.dtypes)
    
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    #print(y_train.dtypes)
    
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    
    y_pred = clf.predict(X_test)
    
    
    cm = confusion_matrix(y_pred,y_test)
    
    # Calcul de l'accuracy à partir de la matrice de confusion
    count_acc = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])


    
    x = df.loc[:,'Message']
    Y = df.loc[:,'Status']
    
    tf = TfidfVectorizer()
    X = tf.fit_transform(x)
    
    X_train, X_test, y_train ,y_test = train_test_split(X,Y,test_size=0.20)
    
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    
    y_pred = clf.predict(X_test)
    
    #print(y_test)
    #print(y_pred)
    
    cm = confusion_matrix(y_pred,y_test)
    #print(cm)
    
    vect_acc = (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])
    
    line = {'Count':count_acc,'TFIDF':vect_acc}
    df_results = df_results.append(line,ignore_index=True)
    
count_mean = df_results.loc[:,'Count'].mean()
tf_mean = df_results.loc[:,'TFIDF'].mean()
print('\nAccuracy CountVectorizer : ',count_mean,'\nAccuracy TF-IDFVectorizer : ',tf_mean)

