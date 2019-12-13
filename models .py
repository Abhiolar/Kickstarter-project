import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus

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



def preprocessing(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size = 0.7)
    stdscaler = StandardScaler()
    stdscaler.fit(X_train)
    X_train_scaled = pd.DataFrame(data = stdscaler.transform(X_train), columns = X_train.columns)
    X_test_scaled = pd.DataFrame(data = stdscaler.transform(X_test), columns = X_test.columns)
    return X_train_scaled, X_test_scaled, y_train, y_test


    
def fit_pred_accuracy(X_train, X_test, y_train, y_test, model):
    
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    probas_trn = model.predict_proba(X_train)
    probas_tst = model.predict_proba(X_test)
    fpr1, tpr1, thresholds1 = roc_curve(y_train, y_pred_train)
    fpr, tpr, thresholds = roc_curve(y_test, probas_tst[:,1])
    fpr1, tpr1, thresholds1 = roc_curve(y_train, probas_trn[:,1])
    confmat=pd.crosstab(y_test, y_pred_test, rownames=['True'], colnames=['Predicted'], margins=True)
    
    print('Train Accuracy Score:',accuracy_score(y_train, y_pred_train), '\n',
          'Test Accuracy Score:',accuracy_score(y_test, y_pred_test),'\n',  
          'AUC train:',auc(fpr1, tpr1),'\n'
          'AUC test:',auc(fpr, tpr),'\n',
           classification_report(y_test, y_pred_test),'\n', confmat)
    
    return fpr, tpr, thresholds, fpr1, tpr1, thresholds1,     

def best_parameters_finder(X_train, X_test, y_train, y_test, model, param_grid, cv, scoring='roc_auc'):
    opt_model = GridSearchCV(model,param_grid,cv=cv,scoring='roc_auc')
    opt_model.fit(X_train,y_train)
    best_model = opt_model.best_estimator_

    y_pred_train=best_model.predict(X_train)
    y_pred_test=best_model.predict(X_test)
    
    probas = best_model.predict_proba(X_test)
    fpr,tpr,thresholds = roc_curve(y_test,probas[:,1])
    
    confmat=pd.crosstab(y_test, y_pred_test, rownames=['True'], colnames=['Predicted'], margins=True)
   
    print('Train Accuracy Score:',accuracy_score(y_train, y_pred_train),'\n',
             'Test Accuracy Score:',accuracy_score(y_test, y_pred_test),'\n', 
            'AUC:',auc(fpr, tpr),'\n',
              opt_model.best_params_, '\n',
             classification_report(y_test, y_pred_test),'\n', confmat)
    return fpr, tpr, thresholds


def optimal_threshold(fpr,tpr,thresholds):
    fpr_cv2 = 3*fpr_cv1
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]


def graph_tree_plot(model, X_train):
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names=X_train.columns)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    
    return Image(graph.create_png())

def plot_roc_curve(fpr, tpr):
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    return