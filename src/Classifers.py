import sklearn.base
import constants
import sklearn.base as base
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from collections import defaultdict

class CustomClassifier(base.BaseEstimator, base.ClassifierMixin):
    AVAILABLE_CLASSIFIERS = {
        'nn':['nn','ANN','DNN'],
        'svm':['svc','svm'],
        'lsvm':['lsvm','linear svm','linsvm'],
        'xgb':['xgb','extragradientboost'],
        'lr':['logistic','logistic regression','lg', 'lr'],
        'gnb':['naive_bayes','naive bayes','gaussian naive bayes','gnb'],
        'pagg':['pagg','passive_aggressive','passive aggressive','passagg','pasag'],
        'ridge':['ridge','rdg','rd'],
        'sgd':['sgd','stochastic gradient descend'],
        'knn':['knn','neighbors','k-nn'],
        'rf':['rf','random forest']
    }

    def __init__(self, class_name: str='nn', hyper_search_type:str='random', scoring:str='balanced_accuracy',
                n_jobs=-1, cv=None, n_iter=200, verbose=3):
        self.est = None
        self._clc_key = None
        self.grid = None
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.scoring=scoring
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
            for name in class_name:
                e, c = self._get_est_key(name)
                self.est[c] = e
            self._isMult_est = True
        else:
            raise ValueError(f"Availabe values for class_name are (str, list or tuple) containing on or more of"
                             f"{self.AVAILABLE_CLASSIFIERS.keys()}")

        self.grid = self._update_grid()
        self.output_models_ = defaultdict(dict)
        self.selector_est_ = None
        self.output_predictions_ = None
        self.output_predictions_log_ = None

    def _handle_estimator(self, est: object) -> object:
        fixed_est = None
        _clc_key = None
        if isinstance(est, str):
            fixed_est, _clc_key = self._get_est_key(est)
            self._ismult_est = False
        elif isinstance(est, (list, tuple)):
            clc_dict = dict()
            for e in est:
                if isinstance(e, str):
                    est, _clc_key = self._get_est_key(e)
                    clc_dict[_clc_key]= est
                else:
                    raise ValueError(f'est should be a string with classifier name, list with classifier names')
            fixed_est = clc_dict
            self._isMult_est=True
        else:
            raise ValueError(f'est should be a string with classifier name, list with classifier names')

        return fixed_est, _clc_key

    def _get_est_key(self, class_name: str) -> tuple:
        est = None
        _clc_key = None
        for key, items_list in self.AVAILABLE_CLASSIFIERS.items():
            if class_name in items_list:
                est = constants.CLC_DICT[key]
                _clc_key=key
                break
        return est, _clc_key

    def _set_single_grid(self, est, key):
        if self.hyper_search_type in 'random':
            grid = RandomizedSearchCV(est, param_distributions=constants.PARAM_GRID[key],scoring=self.scoring,
                                           n_iter=self.n_iter, n_jobs=self.n_jobs, cv=self.cv, verbose=self.verbose)
        elif self.hyper_search_type in 'exhaustive':
            grid = GridSearchCV(est, param_grid=constants.PARAM_GRID[key], scoring=self.scoring,
                                     n_jobs=self.n_jobs, cv=self.cv, verbose=self.verbose)
        else:
            raise ValueError("hyper_search_type can only be either random or exhaustive")
        return grid

    def _update_grid(self):
        if  self._isMult_est:
            self.grid = {}
            for key, est in self.est.items():
                self.grid[key] = self._set_single_grid(est, key)
        else:
            self.grid = self._set_single_grid(self.est, self._clc_key)
        return self.grid

    def set_params(self, **params):
        for key, val in params.items():
            if key == 'est':
                setattr(self, 'est', self._handle_estimator(val)[0])
                setattr(self, '_clc_key', self._handle_estimator(val)[1])
            elif key in self.__dict__.keys():
                setattr(self, key, val)
            else:
                raise ValueError(f'{key} is not a valid CustomClassifier parameter')
        self._update_grid()

    def _fit_single_est(self, X, y, **fit_params):
        self.set_params(**fit_params)
        if isinstance(X, dict):
            for rfe_sel, Xarr in X.items():
                self.output_models_[rfe_sel][self._clc_key] = self.grid.fit(Xarr, y)
        else:
            self.output_models_[self.selector_est_][self._clc_key] = self.grid.fit(X, y)
        # return self.grid
        return self.output_models_

    def _fit_mult_est(self, X, y, **fit_params):
        self.set_params(**fit_params)
        if isinstance(X, dict):
            for rfe_sel, Xarr in X.items():
                for clc in self.grid:
                    self.output_models_[rfe_sel][clc] = self.grid[clc].fit(Xarr, y)
        else:
            for clc in self.grid:
                self.output_models_[self.selector_est_][self._clc_key] = self.grid[clc].fit(X, y)
        # return self.grid
        return self.output_models_

    def fit(self, X, y, **fit_params):
        if isinstance(self.est, dict):
            self.grid = self._fit_mult_est(X, y, **fit_params)
        elif base.is_classifier(self.est):
            self.grid = self._fit_single_est(X, y, **fit_params)
        return self

    def _single_predict(self, X):
        if isinstance(X, dict):
            for rfe_sel, Xarr in X.items():
                self.output_predictions_[rfe_sel][self._clc_key] = self.grid.predict(Xarr)
        else:
            self.output_predictions_[self.selector_est_][self._clc_key] = self.grid.predict(X)
        return self.output_predictions_

    def _mult_predict(self, X):
        if isinstance(X, dict):
            for rfe_sel, Xarr in X.items():
                for clc in self.grid:
                    self.output_predictions_[rfe_sel][clc] = self.grid[clc].predict(Xarr)
        else:
            for clc in self.grid:
                self.output_predictions_[self.selector_est_][clc] = self.grid[clc].predict(X)

        # return results
        return self.output_predictions_

    def _single_predict_proba(self, X):
        if isinstance(X, dict):
            for rfe_sel, Xarr in X.items():
                self.output_predictions_[rfe_sel][self._clc_key] = self.grid.predict_log(Xarr)
        else:
            self.output_predictions_[self.selector_est_][self._clc_key] = self.grid.predict_log(X)
        return self.output_predictions_

    def _mult_predict_proba(self, X):
        if isinstance(X, dict):
            for rfe_sel, Xarr in X.items():
                for clc in self.grid:
                    self.output_predictions_log_[rfe_sel][clc] = self.grid[clc].predict_proba(Xarr)
        else:
            for clc in self.grid:
                self.output_predictions_[self.selector_est_][clc] = self.grid[clc].predict_proba(X)
        return self.output_predictions_log_

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


    def run(self, X, y, **params):
        for key, item in params.items():
            if ('sel' in key) or ('est' in key):
                self.selector_est_ = item
        self.fit(X, y)
        # return self.grid
        return self.output_models_