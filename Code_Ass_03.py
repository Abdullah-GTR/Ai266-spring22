#Mounting Our Drive On Cloab
from google.colab import drive
drive.mount('/content/drive')

#importing Libraries 
import pandas as panda
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt
from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#Importing VarianceThreshold  From Features Selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold, KFold
#Loading Training Data From Drive
train=panda.read_csv('/content/drive/MyDrive/Colab Notebooks/tabular-playground-series-may-2022/train.csv')

#For Labels
y = train.target

#For Features
X = train.drop(['target','f_27'],axis=1)


#Splitting The Data Into 80%(For Training t_train) And 20%(For Testing t_test)
t_train, t_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print(t_train.shape)

# delete rows of duplicate data from the dataset

# delete duplicate rows
t_train.drop_duplicates(inplace=True)

# get number of unique values for each column
counts = train.nunique()
# record columns to delete
to_del = [i for i,v in enumerate(counts) if v == 1]
print(to_del)
# drop useless columns
t_train.drop(to_del, axis=1, inplace=True)
print(t_train.shape)


# delete duplicate rows
t_train.drop_duplicates(inplace=True)
print(t_train.shape)

#Using No Smoothing For Data Fiting,Predicting,And Scoring Accuracy
clf = MultinomialNB(alpha=0)
clf.fit(abs(t_train),y_train)
clf.predict(t_test)
NoMNB=clf.score(t_test,y_test)
print("The Accuracy Score Of NoMNB",NoMNB*100)

skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))

#Using Laplace Smoothing (Alpha=1) For Data Fiting,Predicting,And Scoring Accuracy
clf = MultinomialNB(alpha=1)
clf.fit(abs(t_train),y_train)
clf.fit(abs(t_train),y_train)
clf.predict(t_test)
LPMNB=clf.score(t_test,y_test)
print("The Accuracy Score Of LPMNB",LPMNB*100)
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))

#Using Lidstone  Smoothing (Alpha=0.5) For Data Fiting,Predicting,And Scoring Accuracy
clf = MultinomialNB(alpha=0.5)
clf.fit(abs(t_train),y_train)

clf.predict(t_test)
LDMNB=clf.score(t_test,y_test)
print("The Accuracy Score Of LDMNB",LDMNB*100)

skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))

#Using Perceptron For Data Fiting,Predicting,And Scoring Accuracy
clf = Perceptron(tol=1e-11, random_state=8)
clf.fit(abs(t_train), y_train)
clf.predict(t_test)
Percept=clf.score(t_test,y_test)
print("The Accuracy Score Of Preceptron",Percept*100)

skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))

#Using Support Vector Machine For Data Fiting,Predicting,And Scoring Accuracy
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(abs(t_train), y_train)
clf.predict(t_test)
Svm=clf.score(t_test,y_test)
print("The Accuracy Score Of SVM",Svm*100)

skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))

#Using Knn For Data Fiting,Predicting,And Scoring Accuracy
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(abs(t_train), y_train)
clf.predict(t_test)
Knn=clf.score(t_test,y_test)
print("The Accuracy Score Of Knn",Knn*100)

skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))


#Loading Training Data From Drive
test=panda.read_csv('/content/drive/MyDrive/Colab Notebooks/tabular-playground-series-may-2022/test.csv')
test.head()
test = test.drop('f_27', axis=1)

#Using No Smoothing For Data Fiting,Predicting,And Scoring Accuracy As it's Accuracy Was Good From Others
clf = MultinomialNB(alpha=0)
clf.fit(abs(t_train),y_train)
target=clf.predict(test)
print("The Predicted Values",target)
#define cross-validation method to use
cv = KFold(n_splits=5, random_state=1, shuffle=True) 

#build multiple linear regression model
model = MultinomialNB(alpha=0)

#use LOOCV to evaluate model
scores = cross_val_score(model, abs(t_train), y_train, scoring='neg_mean_squared_error',cv=cv, n_jobs=-1)

#Exporting The Id And Cover_Type Columns Into Sample Csv
sample = test[['id']].copy()
sample['target'] = target
print(sample)

"Creating Our Csv File With That Two Exported Columns For Submission On Kaggle"
sample.to_csv('sample.csv',index=False)
