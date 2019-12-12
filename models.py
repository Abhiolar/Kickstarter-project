import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
import eli5
from eli5.sklearn import PermutationImportance
import lxml


def preprocessing(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = 0.7)
    stdscaler = StandardScaler()
    stdscaler.fit(X_train)
    X_train_scaled = pd.DataFrame(data = stdscaler.transform(X_train), columns = X_train.columns)
    X_test_scaled = pd.DataFrame(data = stdscaler.transform(X_test), columns = X_test.columns)
    return X_train_scaled, X_test_scaled, y_train, y_test

def plot_roc_curve(fpr, tpr):
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    
    
def logregr(X_train, X_test, y_train, y_test):
    logreg = LogisticRegression(fit_intercept=False, C=1e12, solver='liblinear')
    model_log = logreg.fit(X_train, y_train)

    y_pred_test = logreg.predict(X_test)
    y_pred_train = logreg.predict(X_train)
    
    y_score = logreg.fit(X_train, y_train).decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    
    perm = PermutationImportance(logreg, random_state=1).fit(X_test, y_test)
    
    print('Train Accuracy Score:',accuracy_score(y_train, y_pred_train), '\n',
                           'Test Accuracy Score:',accuracy_score(y_test, y_pred_test),'\n', 
                           'Train RocAuc Score:',roc_auc_score(y_train, y_pred_train),'\n',
                           'Test RocAuc Score:',roc_auc_score(y_test, y_pred_test),'\n', 
                           'AUC:',auc(fpr, tpr),'\n',
              classification_report(y_test, y_pred_test))
    
    

    return fpr, tpr, perm  

def decision_tree_model(X_train, X_test, y_train, y_test, max_depth, min_samples_leaf):
    model = DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf)
    model.fit(X_train,y_train)
    probas = model.predict_proba(X_test)
    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)
    
    fpr,tpr, thrsh = roc_curve(y_test,probas[:,1])
    
    perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
    
    print('Train Accuracy Score:',accuracy_score(y_train, y_pred_train),'\n',
         'Test Accuracy Score:',accuracy_score(y_test, y_pred_test),'\n', 
        'Train RocAuc Score:',roc_auc_score(y_train, y_pred_train),'\n', 
        'Test RocAuc Score:',roc_auc_score(y_test, y_pred_test),'\n', 
        'AUC:',auc(fpr, tpr),'\n',
         classification_report(y_test, y_pred_test))
    
    return fpr, tpr, perm

