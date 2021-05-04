import sklearn.base

import constants
import sklearn.base as base
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



class CustomClassifier(base.BaseEstimator, base.ClassifierMixin):
    AVAILABLE_CLASSIFIERS = [
        'nn','svm','lsvm','xgb','lr','gnb','pagg','ridge','sgd','knn','rf'
    ]

    def __int__(self, class_name: str, hyper_search_type:str='random', scoring:str='balanced_accuracy',
                n_jobs=-1, cv=None, n_iter=200, verbose=3):
        self.est = None
        self._clc_key = None
        self.grid = None
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.cv = cv
        self.verbose = verbose
        self.hyper_search_type = hyper_search_type

        if isinstance(class_name, str):
            self.est, self._clc_key = self._get_est_key(class_name)
            if self.est is None:
                raise ValueError(f'Set class_name to be one of the following: {self.AVAILABLE_CLASSIFIERS}')
        elif isinstance(class_name, (list, tuple)):
            self.est = {}
            self._clc_key = []
            for name in class_name:
                e, c = self._get_est_key(name)
                self.est[c] = e
                self._clc_key.append(c)
        else:
            raise ValueError(f"Availabe values for class_name are (str, list or tuple) containing on or more of"
                             f"{self.AVAILABLE_CLASSIFIERS}")

        self.grid = self._update_grid()

    def _get_est_key(self, class_name: str) -> tuple:
        est = None
        _clc_key = None
        if class_name.lower() == 'nn':
            est = MLPClassifier(max_iter=constants.MAX_ITR)
            _clc_key = 'nn'
        elif class_name.lower() == 'svm':
            est = SVC(max_iter=constants.MAX_ITR)
            _clc_key = 'svm'
        elif class_name.lower() == 'lr':
            est = LogisticRegression(max_iter=constants.MAX_ITR)
            _clc_key = 'lr'
        elif class_name.lower() == 'sgd':
            est = SGDClassifier(max_iter=constants.MAX_ITR)
            _clc_key = 'sgd'
        elif class_name.lower() == 'ridge' or class_name.lower() == 'rid' or class_name.lower() == 'rdg':
            est = RidgeClassifier(max_iter=constants.MAX_ITR)
            _clc_key = 'ridge'
        elif class_name.lower() in 'pagg' or class_name.lower() in 'passive' \
                or class_name.lower() in "passiveaggressive":
            est = PassiveAggressiveClassifier(max_iter=constants.MAX_ITR)
            _clc_key = 'pagg'
        elif class_name.lower() == 'knn':
            est = KNeighborsClassifier()
            _clc_key = 'knn'
        elif class_name.lower() in 'naivebayes' or class_name.lower() in 'gnb' or class_name.lower() in 'nb':
            est = GaussianNB()
            _clc_key = 'gnb'
        elif class_name.lower() in 'rf' or class_name.lower() in 'randomforest':
            est = RandomForestClassifier()
            _clc_key = 'rf'
        elif class_name.lower() in 'lsvm' or class_name.lower() in 'linearsvm':
            est = LinearSVC(max_iter=constants.MAX_ITR)
            _clc_key = 'lsvm'
        elif class_name.lower() in 'xgboost':
            est = XGBClassifier()
            _clc_key = 'xgb'

        return est, _clc_key

    def _update_grid(self):
        if self.hyper_search_type in 'random':
            self.grid = RandomizedSearchCV(self.est, param_distributions=constants.PARAM_GRID[self._clc_key],
                                           n_iter=self.n_iter, n_jobs=self.n_jobs, cv=self.cv, verbose=self.verbose)
        elif self.hyper_search_type in 'exhaustive':
            self.grid = GridSearchCV(self.est, param_grid=constants.PARAM_GRID[self._clc_key],
                                     n_jobs=self.n_jobs, cv=self.cv, verbose=self.verbose)
        return self.grid


    def set_params(self, **params):
        for key, val in params.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
            else:
                raise ValueError(f'{key} is not a valid CustomClassifier parameter')
        self._update_grid()

    def set_njobs(self, njobs):
        self.n_jobs = njobs
        self._update_grid()

    def set_cv(self, cv):
        self.cv = cv
        self._update_grid()

    def set_verbose(self, verb):
        self.verbose = verb
        self._update_grid()

    def set_iter(self, iter):
        self.n_iter = iter
        self._update_grid()

    def _fit_single_est(self, X, y, **fit_params):
        self.grid.fit(X, y)
        return self.grid
    def _fit_mult_est(self, X, y, **fit_params):
        self.set_params(fit_params)
        self.grid.fit(X, y)
        return self.grid

    def fit(self, X, y, **fit_params):
        if isinstance(self.est, dict):
            self.grid = self._fit_single_est(X, y, **fit_params)
        elif isinstance(self.est, sklearn.base.ClassifierMixin):
            self.grid = self._fit_mult_est(X, y, **fit_params)

        return self

    def predict(self, X):
        return self.grid.predict(X)

    def predict_proba(self, X):
        return self.grid.predict_proba(X)
