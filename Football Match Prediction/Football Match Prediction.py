import pandas as pd 
import numpy as np 
import statistics 
import seaborn as sns 
import matplotlib.pyplot as plt 
import math 


main = pd.read_csv("C:\\Users\\Admin\\Downloads\\E0.csv") 
df = pd.read_csv("C:\\Users\\Admin\\Downloads\\final.csv")

main.head() 

df.head() 

sns.countplot(x="FTR", data=main) 

sns.countplot(x="Date", hue="FTR", data=main) 

df.describe() 

X_all = df.drop(['FTR'],axis=1) 
y_all = df['FTR'] 


from sklearn.preprocessing import scale 
cols = [['HTGD','ATGD','HTP','ATP','DiffLP']] 
for col in cols: 
    X_all[col] = scale(X_all[col]) 
    

X_all.HM1 = X_all.HM1.astype('str') 
X_all.HM2 = X_all.HM2.astype('str') 
X_all.HM3 = X_all.HM3.astype('str') 
X_all.AM1 = X_all.AM1.astype('str') 
X_all.AM2 = X_all.AM2.astype('str') 
X_all.AM3 = X_all.AM3.astype('str') 

def preprocess_features(X): 
    
    output = pd.DataFrame(index = X.index) 
    
    for col, col_data in X.iteritems(): 
        if col_data.dtype == object: 
            col_data = pd.get_dummies(col_data, prefix = col)

            output = output.join(col_data) 
        return output 
        
X_all = preprocess_features(X_all) 
print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))) 


display(X_all.head()) 


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.5, random_state=2) 


from time import time 
y_preds = [] 
def train_classifier(clf, X_train, y_train): 
    
    # Start the clock, train the classifier, then stop the clock 
    start = time() 
    clf.fit(X_train, y_train) 
    end = time() 
    
def predict_labels(clf, features, target): 
    
    start = time() 
    y_pred = clf.predict(features) 
    y_preds.append(y_pred) 
    end = time() 
    
    
    # Print and return results 
    
    return sum(target == y_pred) / float(len(y_pred))

def train_predict(clf, X_train, y_train, X_test, y_test): 
        print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))) 
        
        train_classifier(clf, X_train, y_train) 
        
        acc = predict_labels(clf, X_train, y_train)
        print("Accuracy score for training set: ", acc*100, "%") 
        
        acc = predict_labels(clf, X_test, y_test) 
        print("Accuracy score for test set: ",acc*100, "%") 
        
        
from sklearn.neural_network import MLPClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC 

mod_A = LogisticRegression(random_state = 30) 
mod_B = MLPClassifier(max_iter= 800,activation='relu',hidden_layer_sizes= 1000) 
mod_c = SVC(random_state = 912) 

train_predict(mod_A, X_train, y_train, X_test, y_test) 
print('') 
train_predict(mod_B, X_train, y_train, X_test, y_test) 
print('') 
train_predict(mod_c, X_train, y_train, X_test, y_test) 
print('') 


print(y_test) 
print(y_preds[1]) 
predss = y_preds[1]


mod_c.predict(X_test)


print('ACTUAL OUTPUT') 
print(' ') 
for i in y_test: 
    if i == 'H': 
        print('Home WIN') 
    elif i == 'NH': 
        print('Away Team win') 
    else: 
        print('Draw') 
print(' ') 
print('PREDICTED OUTPUT') 
print(' ') 
for i in predss: 
    if i == 'H': 
        print('Home WIN') 
    elif i == 'NH': 
        print('Away Team win') 
    else: 
        print('Draw')