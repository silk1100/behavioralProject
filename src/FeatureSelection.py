import sklearn.datasets
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, SelectorMixin
from sklearn.datasets import make_classification
import constants
from sklearn import base
import numpy as np
import os
import dill
from warnings import warn

class FeatureSelector(SelectorMixin, base.BaseEstimator):
    def __init__(self, est='lsvm', selector_method='rfe'):
        self.est = self._handle_estimator(est)
        self.cv=StratifiedKFold(5, shuffle=True, random_state=123)
        self.scoring = 'accuracy'
        self.n_jobs = -1
        self.verbose = 3
        self.step = 1
        self.min_features_to_select = 1
        self.rfe_= None
        self.normalizer = None

        self.selected_feats_ = None
        self.scores_ = None
        self.all_feats_=None

    def _check_valid_selector(self, fixed_est):
        X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=5)
        if isinstance(fixed_est, dict):
            correct_selectors = {}
            for key, sel in fixed_est.items():
                sel.fit(X, y)
                if ('coef_' in sel.__dict__) or ('feature_importances_' in sel.__dict__):
                    correct_selectors[key] = base.clone(sel)
                else:
                    warn(f"{key} can't be used for feature selection")
        else:
            fixed_est.fit(X, y)
            if ('coef_' in fixed_est.__dict__) or ('feature_importances_' in fixed_est.__dict__):
                correct_selectors = base.clone(fixed_est)
            else:
                raise KeyError(f"{fixed_est.__str__()} can't be used for feature selection")
        return correct_selectors

    def _handle_estimator(self, est):
        fixed_est = None
        if isinstance(est, str):
            fixed_est = self._get_clc_from_str(est)
        elif isinstance(est, base.ClassifierMixin):
            fixed_est = est
        elif isinstance(est, (list, tuple)):
            clc_dict = {}
            for e in est:
                if isinstance(e, str):
                    clc_dict[e] = self._get_clc_from_str(e)
                else:
                    raise ValueError(f'est should be a string with classifier name, list with classifier names')
            fixed_est = clc_dict
        else:
            raise ValueError(f'est should be a string with classifier name, list with classifier names, '
                             f'or classifier object')

        fixed_est = self._check_valid_selector(fixed_est)

        return fixed_est

    def _get_est_from_mlobj(self, maindir, mlobj):
        if not isinstance(mlobj, dict):
            raise ValueError(f'ML_obj.p in {maindir} is supposed to be a dictionary')
        if (len(mlobj.keys()) > 1) or (mlobj.get('None') is None):
            raise ValueError(f'ML_obj.p in {maindir} is supposed to have only one key "None" key referring that no FS'
                             f' was conducted')

        classifiers_dict = mlobj['None']
        return classifiers_dict

    def _get_clc_from_str(self, est_name:str):
        if os.path.isdir(est_name):
            file_dir = os.path.join(est_name, 'ML_obj.p')
            with open(file_dir, 'rb') as f:
                ml_obj = dill.load(f)
            class_names_dict = self._get_est_from_mlobj(est_name, ml_obj)
            est = {}
            for key, clc in class_names_dict.items():
                est[key] = base.clone(clc.best_estimator_)
            return est
        else:
            for key, list_names in constants.AVAILABLE_CLASSIFIERS_MAP.items():
                if est_name in list_names:
                    return constants.CLC_DICT[key]()
        raise ValueError(f'estimator name should be one of the following {constants.AVAILABLE_CLASSIFIERS_MAP.keys()}')

    def fit(self, X, y, **rfecv_params):
        if isinstance(self.est, dict):
            self.rfe_ = dict()
            for key, val in self.est.items():
                self.rfe_[key]=RFECV(self.est[key], cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs,
                                       verbose=self.verbose, step=self.step,
                                       min_features_to_select=self.min_features_to_select)
            for key, val in self.rfe_.items():
                self.rfe_[key].fit(X, y)
        else:
            self.rfe_ = RFECV(self.est, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs,
                                       verbose=self.verbose, step=self.step,
                                       min_features_to_select=self.min_features_to_select)
            self.rfe_.fit(X, y)
        return self

    def transform(self, X):
        if self.rfe_ is None:
            raise ValueError("You need to call fit method first")
        if isinstance(self.rfe_, dict):
            if self.normalizer is None:
                return {name: rfe.transform(X) for name, rfe in self.rfe_.items()}
            else:
                return {name: rfe.transform(self.normalizer.transform(X)) for name, rfe in self.rfe_.items()}
        if self.normalizer is None:
            return self.rfe_.transform(X)
        else:
            return self.rfe_.transform(self.normalizer.transform(X))

    def fit_transform(self, X, y=None, **fit_params):
        data_dict = {**self.default_rfecv, **fit_params} if fit_params is not None else self.default_rfecv

        if isinstance(self.est, dict):
            for key, val in self.est.items():
                self.rfe_ = dict()
                self.rfe_[key]=RFECV(self.est[key],**data_dict)
            for key, val in self.rfe_.items():
                self.rfe_[key].fit(X, y)

            return {name: rfe.transform(X) for name, rfe in self.rfe_.items()}

        self.rfe_.fit(X, y)
        return self.rfe_.transform(X)

    def set_params(self, **params):
        for key, val in params.items():
            if key in self.__dict__.keys():
                if key == 'est':
                    setattr(self, 'est', self._handle_estimator(val))
                else:
                    setattr(self, key, val)
            else:
                raise ValueError(f'{key} is not a FeatureSelection valid parameter')

    def _get_support_mask(self):
        if isinstance(self.rfe_, dict):
            return {name: rfe.support_ for name, rfe in self.rfe_.items()}

        return self.rfe_.support_

    def get_support_(self, indices=False):
        if isinstance(self.rfe_, dict):
            if indices:
                return {name: np.where(rfe.support_)[0] for name, rfe in self.rfe_.items()}
            return {name: rfe.support_ for name, rfe in self.rfe_.items()}

        return self.rfe_.support_

    def inverse_transform(self, X):
        if isinstance(self.rfe_, dict):
            return {name: rfe.inverse_transform(X) for name, rfe in self.rfe_.items()}
        return self.rfe_.inverse_transform(X)

    def get_grid_scores(self):
        if isinstance(self.rfe_, dict):
            return {name:rfe.grid_scores_ for name,rfe in self.rfe_.items()}
        return self.rfe_.grid_scores_

    def run(self, X, y=None, normalizer=None, normalize:bool=None):
        if y is None:
            y = X['DX_GROUP']
            X = X.drop('DX_GROUP', axis=1)
        self.all_feats_ = X.columns
        X = X.values
        y = y.values
        if normalize:
            if normalizer is None:
                self.normalizer = StandardScaler()
            else:
                self.normalizer = normalizer

            X = self.normalizer.fit_transform(X)

        self.fit(X, y)
        self.scores_ = self.get_grid_scores()
        if isinstance(self.rfe_, dict):
            self.selected_feats_ = {}
            for key, item in self.get_support().items():
                self.selected_feats_[key] = self.all_feats_[item]
        else:
            self.selected_feats_ = self.all_feats_[self.get_support()]
        return self.transform(X), y, self.normalizer


