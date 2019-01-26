import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pickle
from sklearn.externals import joblib


path1='train_clean.csv'
movies=pd.read_csv(path1,encoding="ISO-8859-1" )

train_x, valid_x, train_y, valid_y = model_selection.train_test_split(movies['review'], movies['sentiment'])

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
print(len(train_y))
#for i in range(0,len(train_y)):
	#print(train_y[i])

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(movies['review'])

xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(movies['review'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

clf = naive_bayes.MultinomialNB()
clf.fit(xtrain_tfidf, train_y)
filename = 'finalized_model.joblib'
filename1 = 'tfifd.joblib'



#pickle.dump(clf, open(filename, 'wb'))
joblib.dump(clf, filename)
joblib.dump(tfidf_vect, filename1)



