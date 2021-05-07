import sklearn.base
import constants
import sklearn.base as base
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class CustomClassifier(base.BaseEstimator, base.ClassifierMixin):
    AVAILABLE_CLASSIFIERS = {
        'nn':['nn','ANN','DNN'],
        'svm':['svc','svm'],
        'lsvm':['lsvm','linear svm','linsvm'],
        'xgb':['xgb','extragradientboost'],
        'lr':['logistic','logistic regression','lg'],
        'gnb':['naive_bayes','naive bayes','gaussian naive bayes','gnb'],
        'pagg':['pagg','passive_aggressive','passive aggressive','passagg','pasag'],
        'ridge':['ridge','rdg','rd'],
        'sgd':['sgd','stochastic gradient descend'],
        'knn':['knn','neighbors','k-nn'],
        'rf':['rf','random forest']
    }

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
        self._isMult_est = False
        if isinstance(class_name, str):
            self.est, self._clc_key = self._get_est_key(class_name)
            if self.est is None:
                raise ValueError(f'Set class_name to be one of the following: {self.AVAILABLE_CLASSIFIERS.keys()}')
        elif isinstance(class_name, (list, tuple)):
            self.est = {}
            self._clc_key = []
            for name in class_name:
                e, c = self._get_est_key(name)
                self.est[c] = e
                self._clc_key.append(c)
            self._isMult_est = True
        else:
            raise ValueError(f"Availabe values for class_name are (str, list or tuple) containing on or more of"
                             f"{self.AVAILABLE_CLASSIFIERS.keys()}")

        self.grid = self._update_grid()

    def _get_est_key(self, class_name: str) -> tuple:
        est = None
        _clc_key = None
        for key, items_list in self.AVAILABLE_CLASSIFIERS:
            if class_name in items_list:
                est = constants.CLC_DICT[key]
                _clc_key=key
                break
        return est, _clc_key

    def _set_single_grid(self, est, key):
        if self.hyper_search_type in 'random':
            grid = RandomizedSearchCV(est, param_distributions=constants.PARAM_GRID[key],
                                           n_iter=self.n_iter, n_jobs=self.n_jobs, cv=self.cv, verbose=self.verbose)
        elif self.hyper_search_type in 'exhaustive':
            grid = GridSearchCV(est, param_grid=constants.PARAM_GRID[self.key],
                                     n_jobs=self.n_jobs, cv=self.cv, verbose=self.verbose)
        else:
            raise ValueError("hyper_search_type can only be either random or exhaustive")
        return grid

    def _update_grid(self):
        if not self._isMult_est:
            self.grid = {}
            for key, est in self.est:
                self.grid[key] = self._set_single_grid(est, key)
        else:
            self.grid = self._set_single_grid(self.est, self._clc_key)
        return self.grid

    def set_params(self, **params):
        for key, val in params.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
            else:
                raise ValueError(f'{key} is not a valid CustomClassifier parameter')
        self._update_grid()

    def _fit_single_est(self, X, y, **fit_params):
        self.set_params(fit_params)
        self.grid.fit(X, y)
        return self.grid

    def _fit_mult_est(self, X, y, **fit_params):
        self.set_params(fit_params)
        for clc in self.grid:
            self.grid[clc].fit(X, y)
        return self.grid

    def fit(self, X, y, **fit_params):
        if isinstance(self.est, dict):
            self.grid = self._fit_single_est(X, y, **fit_params)
        elif isinstance(self.est, sklearn.base.ClassifierMixin):
            self.grid = self._fit_mult_est(X, y, **fit_params)
        return self

    def _single_predict(self, X):
        return self.grid.predict(X)

    def _mult_predict(self, X):
        results = {}
        for clc in self.grid:
            results[clc] = self.grid[clc].predict(X)
        return results

    def _single_predict_proba(self, X):
        return self.grid.predict_proba(X)

    def _mult_predict_proba(self, X):
        results = {}
        for clc in self.grid:
            results[clc] = self.grid[clc].predict_proba(X)
        return results

    def predict(self, X):
        if self._isMult_est:
            result = self._single_predict(X)
        else:
            result = self._mult_predict(X)
        return result

    def predict_proba(self, X):
        if self._isMult_est:
            result = self._single_predict_proba(X)
        else:
            result = self._mult_predict_proba(X)
        return result
