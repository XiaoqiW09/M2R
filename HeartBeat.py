import numpy as np
import iisignature as sig
from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from skopt import BayesSearchCV
from skopt.space.space import Real, Integer, Categorical
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt


from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn_genetic.plots import plot_fitness_evolution, plot_search_space

X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("Heartbeat")

X_train_new = TimeSeriesScalerMeanVariance().fit_transform(X_train)
X_test_new = TimeSeriesScalerMeanVariance().fit_transform(X_test)

arr = sig.sig(X_train_new, 2)
arr_test = sig.sig(X_test_new, 2)

scaler = StandardScaler()
arr_new = scaler.fit_transform(arr)
arr_test_new = scaler.fit_transform(arr_test)

# svc = SVC(probability=True, cache_size=400, random_state=0)
# parameters = {'gamma': ['scale', 'auto'], 'degree': range(1,6), 'C': [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10],
#               'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
# clf = GridSearchCV(svc, parameters, scoring='roc_auc', n_jobs=10)
# print(clf)

boost = GradientBoostingClassifier(loss='log_loss')
parameters = {'learning_rate': [0.1, 0.2, 0.5, 1, 2, 4, 8],
              'n_estimators': [100, 150, 200, 250, 300],
              'subsample': [0.25, 0.5, 0.75, 1]}
clf = GridSearchCV(boost, parameters, scoring='roc_auc', n_jobs=10, verbose=True)
clf.fit(arr_new, y_train)
res = roc_auc_score(y_test, clf.predict_proba(arr_test_new)[:, 1])
print(res)
