import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pickle
from sklearn.externals import joblib
import csv

filename = 'finalized_model.joblib'
filename1 = 'tfifd.joblib'
#clf= pickle.load(open(filename, 'rb'))
clf = joblib.load(filename)
tfidf_vect = joblib.load(filename1)


#tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
mi=pd.read_csv("mi_clean.csv",encoding="ISO-8859-1" )


mir =  tfidf_vect.transform(mi['text'])
prediction_result = clf.predict(mir)
#for i in range(0,len(prediction_result)):
#	print(mi[i])
#	print(prediction_result[i])


i=0
with open('mi_clean.csv', 'r',encoding="ISO-8859-1") as csvfile:
    with open('mi_clean_result.csv', 'w',encoding="ISO-8859-1",newline='') as csvfilew:
        fp = csv.reader(csvfile, delimiter=',')
        fpw=csv.writer(csvfilew,delimiter=',')
        header = next(fp)
        for row in fp:
            
            fpw.writerow([row[0], prediction_result[i]])
            i+=1

print('Result saved')
print('enter review')
sen=input()
yaa = np.array([[sen]])
ngran= tfidf_vect.transform(yaa.ravel())
res=clf.predict(ngran)
if res[0]:
	print('positive')
else:
	print('negative')

'''import csv
with open('C:/Users/hp/Desktop/mi_clean_result.csv', 'r',encoding="ISO-8859-1") as csvfile:
    fp = csv.reader(csvfile, delimiter=',')
    count1=0
    count0=0
    for row in fp:
        sen=row[1]
        if sen=="1":
            count1+=1
        else:
            count0+=1


data = [['pos',count1],['neg',count0]]
df = pd.DataFrame(data,columns=['sen_type','sentiment'])
sen_df=df.set_index('sen_type')

sen_df.plot(kind='bar',title='Twitter Sentiment Analysis ')
plt.show()'''