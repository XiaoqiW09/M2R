import numpy as np
import iisignature as sig
from tslearn.datasets import UCR_UEA_datasets
from sklearn import svm
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# load data
X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset("EyesOpenShut")

# initialise empty matrices to populate with signatures
arr = np.zeros((len(y_train),210))
arr_test = np.zeros((len(y_test),210))

# change each time series matrix into a signature array at level 2
for i in range(len(X_train)):
    arr[i] = sig.sig(X_train[i], 2)
for i in range(len(X_test)):
    arr_test[i] = sig.sig(X_test[i], 2)

# model testing
clf = LogisticRegression(solver="liblinear", random_state=0)
clf.fit(arr, y_train)
res = roc_auc_score(y_test, clf.predict_proba(arr_test)[:, 1])
print(res)
parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':range(1,11), 'gamma': ('scale', 'auto')}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, scoring='accuracy')
clf.fit(arr, y_train)
sorted(clf.cv_results_.keys())
res = accuracy_score(y_test, clf.predict(arr_test))
print(res)