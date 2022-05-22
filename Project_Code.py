from google.colab import drive
drive.mount('/content/drive')

#importing Libraries 
import pandas as panda
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from numpy import mean
from numpy import absolute
from numpy import sqrt
from sklearn.linear_model import Perceptron
import numpy as np
#Importing MultiClass Classifier DecisionTreeClassifier From SkLearn
from sklearn.tree import DecisionTreeClassifier
#Importing MultiClass Classifier BaggingClassifier From SkLearn
from sklearn.ensemble import BaggingClassifier
#Importing MultiClass Classifier ExtraTreeClassifier From SkLearn
from sklearn.tree import ExtraTreeClassifier
#Importing MultiClass Classifier RidgeClassifier From SkLearn
from sklearn.linear_model import RidgeClassifier
#Importing MultiClass Classifier RandomForestClassifier From SkLearn
from sklearn.ensemble import RandomForestClassifier
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
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(abs(t_train), y_train)
clf.predict(t_test)
Percept=clf.score(t_test,y_test)
print("The Accuracy Score Of Preceptron",Percept*100)

skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
  print('train -  {}   |   test -  {}'.format(np.bincount(y[train]), np.bincount(y[test])))

#Abdur Rehman(10875), Mohammad Huzaifa (10830) Part
#Here We Have Tweaked Our classifiers By  Adjusting The Parameters
extra_tree = ExtraTreeClassifier(random_state=0,criterion ='entropy')
cls = BaggingClassifier(extra_tree, random_state=0).fit(t_train, y_train)
#Fitting Our training Data
cls = BaggingClassifier(extra_tree, random_state=0).fit(t_train, y_train)
#Predicting The Values Of Local Splitted Test Data
Cover_type = cls.predict(t_test)
#Printing The Predicted Value Of Local Splitted Test Data
print("The Predicted Values Cover Type Of Local Test Data",Cover_type)
#Checking The  accuracy This Model Using The Local Test Data 
ETree=cls.score(t_test,y_test)
#Printing The Accuracy Score Of The Decision Tree Classifier 
print("The Accuracy Score Of Extra Tree Classifier On Local Test Data",ETree*100)

#Abdur Rehman(10875), Mohammad Huzaifa (10830) Part
#Here We Are #Fitting Our training Data
clf = RidgeClassifier().fit(t_train, y_train)
#Predicting The Values Of Local Splitted Test Data
Cover_type = clf.predict(t_test)
#Printing The Predicted Value Of Local Splitted Test Data
print("The Predicted Values Cover Type Of Local Test Data",Cover_type)
#Checking The  accuracy This Model Using The Local Test Data 
Ridge=clf.score(t_test,y_test)
#Printing The Accuracy Score Of The Decision Tree Classifier 
print("The Accuracy Score Of RidgeClassifier On Local Test Data",Ridge*100)

#Ahmer Hussin(10735), Mohammad Faiz(10637) Part
#Here We Are #Fitting Our training Data
clf = RandomForestClassifier(max_depth=100, random_state=0)
clf.fit(t_train,y_train)
#Predicting The Values Of Local Splitted Test Data
Cover_type = clf.predict(t_test)
#Printing The Predicted Value Of Local Splitted Test Data
print("The Predicted Values Cover Type Of Local Test Data",Cover_type)
#Checking The  accuracy This Model Using The Local Test Data 
RandomForest=clf.score(t_test,y_test)
#Printing The Accuracy Score Of The Decision Tree Classifier 
print("The Accuracy Score Of RidgeClassifier On Local Test Data",RandomForest*100)
#Loading Training Data From Drive
test=panda.read_csv('/content/drive/MyDrive/Colab Notebooks/tabular-playground-series-may-2022/test.csv')
test.head()
test = test.drop('f_27', axis=1)

#Ahmer Hussin(10735), Mohammad Faiz(10637) Part
#Here We Have Tweaked Our classifiers By  Adjusting The Parameters
dtree_model = DecisionTreeClassifier(criterion="entropy",max_depth = 20000000).fit(t_train, y_train)
#Predicting The Values Of Local Splitted Test Data
Target = dtree_model.predict(t_test)
#Printing The Predicted Value Of Local Splitted Test Data
print("The Predicted Values Cover Type Of Local Test Data",Target)
#Checking The  accuracy This Model Using The Local Test Data
Tree=dtree_model.score(t_test,y_test)
#Printing The Accuracy Socre Of The Decision Tree Classifier
print("The Accuracy Score Of Decision Tree Classifier On Local Test Data",Tree*100)
#Predicting The Test Data Values
Target=dtree_model.predict(test)
#Printing The Predicting Values
print("The Predicted Values For The Kaggle Test Data Cover Type",Target)
#Exporting The Id And Cover_Type Columns Into Sample Csv
sample = test[['id']].copy()
sample['target'] = Target
print(sample)
#Creating Our Csv File With That Two Exported Columns For Submission On Kaggle
sample.to_csv('AI_Project.csv',index=False)


