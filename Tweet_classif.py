#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis pour des tweets :
# 
# Ce document explore plusieurs méthodes qui permettent de prédire le sentiment général des tweets et les classifier en 3 classes : Négatif, Neutre et Positif.
# 
# On peut adapter ce traitement à des articles en francais. On peut entraîner un modèle sur un langage similaire à celui présent dans les revues de presse.
# 
# On peut aussi utiliser des méthodes de Deep Learning (deep convolutional neural networks), même si ce genre de modèle a une mauvaise performance s'il est entraîné sur un petit nombre d'observations.



import numpy as np
import nltk
nltk.download('punkt')
import pandas as pd
import matplotlib .pyplot as plt 


# Récupération des données 

#Train 


path=r'C:/Users/Nicole/Desktop/Stage code/Dataset/train.csv'
df_train=pd.read_csv(path)
df_train['sentiment'].replace({'neutral':0,'positive':1,'negative':-1},inplace=True) 

#compte nombre sentiment positif, negatif , neutre
df_train['sentiment'].value_counts()

# graphqiue sentiment
Sentiment_count=df_train.groupby('sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['selected_text'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.xticks([-1,0,1], ["Negative","Neutral", "Positive"])
plt.show()


#df_train.head(10)


#Test


path=r'C:/Users/Nicole/Desktop/Stage code/Dataset/test.csv'
df_test=pd.read_csv(path)
df_test['sentiment'].replace({'neutral':0,'positive':1,'negative':-1},inplace=True)

nb_test=df_test.shape[0]
##########

df=pd.concat((df_train,df_test))

#renommer indice
df.selected_text[df.selected_text.isna()]=df.text
df.reset_index(inplace=True)
df.drop('index',axis=1,inplace=True)
df.head(10)


#Pre-processing 

# In[312]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet')


df.dtypes


# conversion des float en string 
df['selected_text']=df['selected_text'].apply(lambda x: str(x)) 

# conversion en minuscule
df['selected_text']=df['selected_text'].apply(lambda x: str.lower(x))

#stopwords
stop = stopwords.words('english')
df['selected_text'].apply(lambda x: [item for item in x if item not in stop])
df.head(5)


#Lemmatization
# 
# Lemmatization est une autre technique plus populaire avec plusieurs modèles pré-entraînés. (WordNet disponible sur nltk)



wordnet_lemmatizer = WordNetLemmatizer()
nrows = len(df)
lemmatized_text_list = []
tokenizer = nltk.RegexpTokenizer(r"\w+")

for row in range(0, nrows):
   # Save the text and its words into an object
    text = str(df.loc[row]['text'])
    text = text.lower()
    text.strip()
    text_words = tokenizer.tokenize(text)
    text_words=[word for word in text_words if word not in stop]
    text_words=[word for word in text_words if '@' not in word]
    text_words=[word for word in text_words if '#' not in word]
    text_words=[word for word in text_words if 'http' not in word]
    lemmatized_text=" ".join([wordnet_lemmatizer.lemmatize(word, pos="v") for word in text_words])
    
    # Append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)


df['lemmas'] = lemmatized_text_list
df.head(5)


# Text representation


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import make_pipeline


#TF-IDF vectors

# $$ TFIDF(t,d)=TF(t,d)log(\frac{N}{DF(t)})$$
# $$ TF(t,d) = \frac{Nombre~d'apparition~de~ t~ dans~ d}{nombre~ total~ de~ mots~ dans~ d}$$
# $$ DF(t) = nombre~ de~ documents~ contenant ~t$$



X_train, X_test, y_train, y_test = train_test_split(df[["lemmas"]], df['sentiment'], test_size=nb_test, shuffle=False)

pipe = make_pipeline(CountVectorizer(), TfidfTransformer())

pipe.fit(X_train['lemmas'])

feat_train_tfidf = pipe.transform(X_train['lemmas'])

feat_test_tfidf = pipe.transform(X_test['lemmas'])

#################
# Word embeddings 

# C'est une représentation apprise d'un texte où les mots qui ont la même signification ont une représentation similaire.
# $f(d)$ est un représentant de $d$ dans un nouvel espace, le choix de ce nouvel espace est effectué de telle sorte à ce que $f(d)$ ne soit pas "sparse".
# 
# ""The distributed representation is learned based on the usage of words. This allows words that are used in similar ways to result in having similar representations, naturally capturing their meaning.""
# 
# Le fait qu'un grand nombre de tweets dans cet exercice comporte plusieurs fautes d'orthographe (donc plusieurs irrégularités pour les mots) rendera l'apprentissage du word embedding plus difficile.
# 
# Pour entraîner le word embedding, on a deux approches possibles, la première est le CBOW qui cherche à apprendre l'entourage  à partir du mot, la deuxième est le Skip-gram et consiste à apprendre le mot à travers son entourage.
# 
# on utilise word2Vec : https://pathmind.com/wiki/word2vec#:~:text=Word2vec%20is%20a%20two%2Dlayer,deep%20neural%20networks%20can%20understand.

# Entrainer le Word2Vec

# On va entraîner le Word2Vec en utilisant une base de données que j'ai trouvé sur internet contenant 1.6 millions tweets, cela servira seulement comme exemple illustratif ce que l'on peut faire avec cet outil.




import gensim
from gensim.models.word2vec import Word2Vec


path=r'C:/Users/Nicole/Desktop/Stage code/Dataset/tweets.csv'
extra_tweets=pd.read_csv(path,encoding='latin-1',usecols=[5])
extra_tweets.columns = ['tweet']
extra_tweets.head(5)


#preprocessing 
sentences=list(extra_tweets.tweet)
tokenizer = nltk.RegexpTokenizer(r"\w+")

for i in range(len(sentences)) :
    s=str(sentences[i])
    s=s.lower()
    s.strip() #remove extra whitespaces
    s=[word for word in s.split(' ') if word not in stop] #on enleve les stopwords et on transforme la liste en string
    s=[word for word in s if '@' not in word]
    s=[word for word in s if '#' not in word]
    s=[word for word in s if 'http' not in word]
    s=[word for word in s if word!='']
    s=" ".join([wordnet_lemmatizer.lemmatize(word, pos="v") for word in s])
    s = tokenizer.tokenize(s)
    sentences[i]=s


#notre propre word embedding
#Skip-gram
size_=50
model = Word2Vec(sentences, min_count=1,size=size_,workers=3, window =3, sg = 1) #sg={0,1} 1=skip-ram 0=CBOW



tweets=[ " ".join(s) for s in sentences]


# Resultats 
# On montre quelques résultats du modèle. Most_similar(x) nous donne les mots les plus similaire à x (qui ont une représentation vectorielle colinéaire à x), et le chiffre affiché devant représente la similarité entre les deux mots.



model.most_similar('good')
model.most_similar('work')
model.most_similar('unemployed')
model.most_similar('crisis')
model.most_similar('kid')


model['kid']

#######################
# Feature Engineering 

#Poids = TfIdf

# Avec le modèle de Word Embedding, chaque mot est représenté par un vecteur $x=(x_1,...,x_n)$, donc pour représenter une phrase on peut combiner linéairement tout ces vecteurs à l'aide de certains poids.
# 
# Si on prend comme poids $poids= \frac{1}{longueur~ de~ la~ phrase}$, 'the' aura le même apport qu'un autre mot plus significatif. Si on prend comme poids les score TD-IDF on va favoriser les mots qui ont moins d'occurences.
# 
# 
# Afin d'utiliser les TD-IDF de facon efficace, on va créer une matrice 2D.



cv=CountVectorizer()
#le vocabulaire était celui du train avec cette commande ce qui crééer des problèmes avec avec le word embedding par la suite
#word_count_vector=cv.fit_transform(list(X_train['lemmas'])) 
word_count_vector=cv.fit_transform(tweets)

count_vector=cv.transform(list(X_train['lemmas']))
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
tf_idf_matrix=tfidf_transformer.transform(count_vector)


print(tf_idf_matrix[0])
print(tf_idf_matrix[0].data)



feat_train=[]
for i in range(X_train.shape[0]):
    arr=np.zeros(size_)
    sentence=X_train.iloc[i].lemmas
    words=sentence.split(' ')
    weights=tf_idf_matrix[i].data
    words=list(dict.fromkeys(words))
    k=0
    for j in range(len(words)):
        if words[j] in model.wv.vocab and len(words[j])>1:
            arr+= model[words[j]]*weights[k]
            k+=1
    feat_train.append(arr)


count_vector_test=cv.transform(list(X_test['lemmas']))
tf_idf_matrix_test=tfidf_transformer.transform(count_vector_test)

feat_test=[]
for i in range(X_test.shape[0]):
    arr=np.zeros(size_)
    sentence=X_test.iloc[i].lemmas
    words=sentence.split(' ')
    weights=tf_idf_matrix_test[i].data
    words=list(dict.fromkeys(words))
    k=0
    for j in range(len(words)):
        if words[j] in model.wv.vocab and len(words[j])>1:
            arr+= model[words[j]]*weights[k]
            k+=1
        
    feat_test.append(arr)

######################
# Entrainement du modèle de prédiction


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


#Random Forest

clf = RandomForestClassifier(n_estimators=50,max_depth=15,bootstrap=True,random_state=1,criterion='entropy')
clf.fit(feat_train, y_train)

predicted=clf.predict(feat_test)


print(confusion_matrix(y_test, predicted))
target_names = ['class -1', 'class 0', 'class 1']
print(classification_report(y_test, predicted, target_names=target_names))


#logistic Regression 
clf = LogisticRegression(random_state=3, penalty ='l1',C=0.05)
clf.fit(feat_train, y_train)

predicted=clf.predict(feat_test)


print(confusion_matrix(y_test, predicted))
target_names = ['class -1', 'class 0', 'class 1']
print(classification_report(y_test, predicted, target_names=target_names))


#SVM avec noyau RBF
clf = SVC(C=10,kernel='rbf')
clf.fit(feat_train, y_train)

predicted=clf.predict(feat_test)



print(confusion_matrix(y_test, predicted))
target_names = ['class -1', 'class 0', 'class 1']
print(classification_report(y_test, predicted, target_names=target_names))


# Poids uniforme 


feat_train=[]
for i in range(X_train.shape[0]):
    arr=np.zeros(size_)
    sentence=X_train.iloc[i].lemmas
    words=sentence.split(' ')
    words=list(dict.fromkeys(words))
    for j in range(len(words)):
        if words[j] in model.wv.vocab :
            arr+= model[words[j]]/len(words)
    feat_train.append(arr)


count_vector_test=cv.transform(list(X_test['lemmas']))
tf_idf_matrix_test=tfidf_transformer.transform(count_vector_test)

feat_test=[]
for i in range(X_test.shape[0]):
    arr=np.zeros(size_)
    sentence=X_test.iloc[i].lemmas
    words=sentence.split(' ')
    words=list(dict.fromkeys(words))
    for j in range(len(words)):
        if words[j] in model.wv.vocab :
            arr+= model[words[j]]/len(words)
        
    feat_test.append(arr)


#Prédiction 



#Random Forest

clf = RandomForestClassifier(n_estimators=50,max_depth=15,bootstrap=True,random_state=1,criterion='entropy')
clf.fit(feat_train, y_train)

predicted=clf.predict(feat_test)


# In[356]:


print(confusion_matrix(y_test, predicted))
target_names = ['class -1', 'class 0', 'class 1']
print(classification_report(y_test, predicted, target_names=target_names))


#logistic Regression 
clf = LogisticRegression(random_state=3, penalty ='l1',C=0.05)
clf.fit(feat_train, y_train)

predicted=clf.predict(feat_test)


print(confusion_matrix(y_test, predicted))
target_names = ['class -1', 'class 0', 'class 1']
print(classification_report(y_test, predicted, target_names=target_names))


#K-NN

clf = KNeighborsClassifier(n_neighbors=10,p=1, metric='minkowski') #problème de métrique en grande dimension ?????
clf.fit(feat_train, y_train)

predicted=clf.predict(feat_test)


print(confusion_matrix(y_test, predicted))
target_names = ['class -1', 'class 0', 'class 1']
print(classification_report(y_test, predicted, target_names=target_names))


#SVM avec noyau RBF
clf = SVC(C=10,kernel='rbf')
clf.fit(feat_train, y_train)

predicted=clf.predict(feat_test)


print(confusion_matrix(y_test, predicted))
target_names = ['class -1', 'class 0', 'class 1']
print(classification_report(y_test, predicted, target_names=target_names))


#  Prediction avec TF-IDF uniquement


clf = LogisticRegression(random_state=3, penalty ='l1',C=0.05)
clf.fit(feat_train_tfidf, y_train)

predicted=clf.predict(feat_test_tfidf)


print(confusion_matrix(y_test, predicted))
target_names = ['class -1', 'class 0', 'class 1']
print(classification_report(y_test, predicted, target_names=target_names))


#Entraînement non supervisée (sans les labels 0 1 et -1)

# Une possibilité d'analyse des sentiments en absence des labels, serait d'utiliser des algorithmes de clustering pour séparer les termes à connotation positive de ceux à connotation négative.
# 
# Une deuxième possibilité est d'utiliser des méthodes d'apprentissage semi-supervisé. 
# 
# On commence par une visualisation des trois classes.


# from sklearn.manifold import TSNE
# tsne_model=TSNE(n_components=2, verbose=1, random_state=0)
# tsne_w2v=tsne_model.fit_transform(feat_train)
# 
# tsne_w2v.shape
# 
# 
# import seaborn as sns
# 
# 
# 
# color_dict = dict({-1:'red', 0:'green', 1:'blue'})
# data_=pd.DataFrame(data=tsne_w2v, columns=['Axe1','Axe2'])
# 
# data_['sentiment']=y_train
# data_.head(5)
# 
# 
# g=sns.scatterplot(x="Axe1", y="Axe2",hue='sentiment',data=data_,palette=color_dict,legend='full')
# g.set(xscale="log")
# #-------------------
# 
# tsne_model=TSNE(n_components=2, verbose=1, random_state=0)
# tsne_w2v=tsne_model.fit_transform(feat_test)
# 
# 
# color_dict = dict({-1:'red', 0:'green', 1:'blue'})
# data_=pd.DataFrame(data=tsne_w2v, columns=['Axe1','Axe2'])
# 
# data_['sentiment']=list(y_test)
# data_.head(5)
# 
# 
# 
# g=sns.scatterplot(x="Axe1", y="Axe2",hue='sentiment',data=data_,palette=color_dict,legend='full')
# g.set(xscale="log")
