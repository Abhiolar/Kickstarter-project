import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score,roc_curve,auc ,confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from IPython.display import Image  
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import eli5
from eli5.sklearn import PermutationImportance
import numpy as np





def distribution_check(kick):
    fig = plt.figure(figsize = (20,10))

    ax1 = fig.add_subplot(2,3,1)
    ax2 = fig.add_subplot(2,3,2)
    ax3 = fig.add_subplot(2,3,3)
    ax4 = fig.add_subplot(2,3,4)
  

    sns.distplot(kick['usd_pledged_real'], color = 'red', ax = ax1)
    sns.distplot(kick['usd_goal_real'], color = 'blue', ax = ax2)
    sns.distplot(kick['number_of_days'], color = 'green', ax = ax3)
    sns.distplot(kick['backers'], color = 'purple', ax = ax4)