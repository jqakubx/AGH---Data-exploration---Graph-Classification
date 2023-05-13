import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score


def evaluate_random_forest(X, Y, n_splits, n_eval = 10, **rf_kwargs):
    cvs_accs = []
    cvs_f1s = []
    n = n_eval
    for i in range(n):
        # after grid search, the best parameter is {'kernel': 'rbf', 'C': 100, 'gamma': 0.1}
        clf = RandomForestClassifier(**rf_kwargs)

        k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
        cvs_acc = cross_val_score(clf, X, Y, n_jobs=-1, cv=k_fold, scoring='accuracy')
        cvs_f1 = cross_val_score(clf, X, Y, n_jobs=-1, cv=k_fold, scoring='f1_weighted')
        # print(cvs)
        acc = cvs_acc.mean()
        f1 = cvs_f1.mean()
        cvs_accs.append(acc)
        cvs_f1s.append(f1)
    accuracy = np.array(cvs_accs)
    f1 = np.array(cvs_f1s)
    print('Cross val score accuracy: %s' % accuracy.mean())
    print('Cross val score f1: %s' % f1.mean())

    cvs_accs = []
    cvs_f1s = []
    for i in range(n):

        clf = RandomForestClassifier(**rf_kwargs)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        cvs_accs.append(acc)
        cvs_f1s.append(f1)
    accuracy = np.array(cvs_accs)
    f1 = np.array(cvs_f1s)
    print("Holdout Accuracy: %s" % accuracy.mean())
    print('Holdout F1 score: %s' % f1.mean())
    return (accuracy.mean(), accuracy.std())
