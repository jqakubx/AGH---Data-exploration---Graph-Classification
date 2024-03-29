import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import laplacian_kernel

PRINT_TIME = False

def timefunction(method, time_flag=False):
    def timed(*args, **kw):
        import time
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            if (te-ts)>1 and PRINT_TIME:
                print('%r  %2.2f s' % (method.__name__, (te - ts) ))
        if time_flag == False:
            return result
        else:
            return result, te-ts
    return timed

@timefunction
def search_params_svm(X, Y, i, test_size=0.1, nonlinear_flag='False', verbose=0, print_flag='off'):
    if nonlinear_flag == 'True':
        tuned_parameters = [{'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000]},
                            {'kernel': ['rbf'], 'gamma': [0.01, 0.1, 1, 10,100], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    elif nonlinear_flag == 'False':
        tuned_parameters = [{'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    elif nonlinear_flag == 'reddit_12K': # reddit_12K is slow
        tuned_parameters = [{'kernel': ['linear'], 'C': [1000]},
                            {'kernel': ['rbf'], 'gamma': [1, 10, 100 ], 'C': [1000]}]
    else:
        raise Exception('Unconsidered case for nonlinear_flag in searchclf')

    for score in ['accuracy']:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=i)
        clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=10, scoring='%s' % score, n_jobs=-1, verbose=verbose)
        clf.fit(X_train, y_train)

        if print_flag=='on':
            print("Best parameters set found on development set is \n %s with score %s" % (clf.best_params_, clf.best_score_))
            print(clf.best_params_)
            print("Grid scores on development set:\n")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        if print_flag == 'on':
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
            print("Detailed classification report:\n")
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            print()
            y_true, y_pred = y_test, clf.predict(X_test)
            print(classification_report(y_true, y_pred))
        return clf.best_params_

@timefunction
def evaluate_svm(X, Y, best_params_, n_splits, n_eval = 10):
    cvs_accs = []
    cvs_f1s = []
    n = n_eval
    for i in range(n):
        # after grid search, the best parameter is {'kernel': 'rbf', 'C': 100, 'gamma': 0.1}
        if best_params_['kernel'] == 'linear':
            clf = svm.SVC(kernel='linear', C=best_params_['C'])
        elif best_params_['kernel'] == 'rbf':
            clf = svm.SVC(kernel='rbf', C=best_params_['C'], gamma=best_params_['gamma'])
        elif best_params_['kernel'] == 'precomputed': # take care of laplacian case
            clf = svm.SVC(kernel='precomputed', C=best_params_['C'])
        else:
            raise Exception('Parameter Error')

        k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=i)
        if clf.kernel == 'precomputed':
            laplacekernel = laplacian_kernel(X, X, gamma=best_params_['gamma'])
            cvs_acc = cross_val_score(clf, laplacekernel, Y, n_jobs=-1, cv=k_fold, scoring='accuracy')
            cvs_f1 = cross_val_score(clf, laplacekernel, Y, n_jobs=-1, cv=k_fold, scoring='f1')
            print('CV Laplacian kernel')
        else:
            cvs_acc = cross_val_score(clf, X, Y, n_jobs=-1, cv=k_fold, scoring='accuracy')
            cvs_f1 = cross_val_score(clf, X, Y, n_jobs=-1, cv=k_fold, scoring='f1')
        # print(cvs)
        acc = cvs_acc.mean()
        f1 = cvs_f1.mean()
        cvs_accs.append(acc)
        cvs_f1s.append(f1)
    accuracy = np.array(cvs_accs)
    f1 = np.array(cvs_f1s)
    print('Cross val score accuracy: %.4f' % accuracy.mean())
    print('Cross val score f1: %.4f' % f1.mean())

    cvs_accs = []
    cvs_f1s = []
    for i in range(n):

        if best_params_['kernel'] == 'linear':
            clf = svm.SVC(kernel='linear', C=best_params_['C'])
        elif best_params_['kernel'] == 'rbf':
            clf = svm.SVC(kernel='rbf', C=best_params_['C'], gamma=best_params_['gamma'])
        elif best_params_['kernel'] == 'precomputed':  # take care of laplacian case
            clf = svm.SVC(kernel='precomputed', C=best_params_['C'])
        else:
            raise Exception('Parameter Error')

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cvs_accs.append(acc)
        cvs_f1s.append(f1)
    accuracy = np.array(cvs_accs)
    f1 = np.array(cvs_f1s)
    print("Holdout Accuracy: %.4f" % accuracy.mean())
    print('Holdout F1 score: %.4f' % f1.mean())
