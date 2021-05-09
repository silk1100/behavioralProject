from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, SelectorMixin
import constants
from sklearn import base

class FeatureSelector(SelectorMixin, base.BaseEstimator):
    def __init__(self, est=constants.CLC_DICT['lsvm'], selector_method='rfe'):
        self.est = est
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

    def fit(self, X, y, **rfecv_params):
        if isinstance(self.est, (list, tuple)):
            for idx, e in enumerate(self.est):
                self.rfe_ = list()
                self.rfe_.append(RFECV(self.est[idx], cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs,
                                       verbose=self.verbose, step=self.step,
                                       min_features_to_select=self.min_features_to_select))
            for idx, rfe in enumerate(self.rfe_):
                self.rfe_[idx].fit(X, y)
        else:
            self.rfe_ = RFECV(self.est, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs,
                                       verbose=self.verbose, step=self.step,
                                       min_features_to_select=self.min_features_to_select)
            self.rfe_.fit(X, y)
        return self

    def transform(self, X):
        if self.rfe_ is None:
            raise ValueError("You need to call fit method first")
        if isinstance(self.rfe_, list):
            return [rfe.transform(X) for rfe in self.rfe_]

        return self.rfe_.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        data_dict = {**self.default_rfecv, **fit_params} if fit_params is not None else self.default_rfecv

        if isinstance(self.est, (list, tuple)):
            for idx, e in enumerate(self.est):
                self.rfe_ = list()
                self.rfe_.append(RFECV(self.est[idx], **data_dict))
            for idx, rfe in enumerate(self.rfe_):
                self.rfe_[idx].fit(X, y)

            return [rfe.transform(X) for rfe in self.rfe_]

        self.rfe_.fit(X, y)
        return self.rfe_.transform(X)

    def set_params(self, **params):
        for key, val in params.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
            else:
                raise ValueError(f'{key} is not a FeatureSelection valid parameter')

    def _get_support_mask(self):
        if isinstance(self.rfe_, (list, tuple)):
            return [rfe._get_support_mask() for rfe in self.rfe_]

        return self.rfe_.get_support_mask()

    def get_support_(self, indices=False):
        if isinstance(self.rfe_, (list, tuple)):
            return [rfe.get_support(indices) for rfe in self.rfe_]

        return self.rfe_.get_support(indices)

    def inverse_transform(self, X):
        if isinstance(self.rfe_, (list, tuple)):
            return [rfe.inverse_transform(X) for rfe in self.rfe_]
        return self.rfe_.inverse_transform(X)

    def get_grid_scores(self):
        if isinstance(self.rfe_, (list, tuple)):
            return [rfe.grid_scores_ for rfe in self.rfe_]
        return self.rfe_.grid_scores_

    def run(self, X, y=None, normalize=True):
        if y is None:
            y = X['DX_GROUP']
            X = X.drop('DX_GROUP', axis=1)
        self.all_feats_ = X.columns
        X = X.values
        y = y.values
        if normalize:
            if self.normalizer is None:
                self.normalizer = StandardScaler()
            X = self.normalizer.fit_transform(X)
        self.fit(X, y)
        self.scores_ = self.get_grid_scores()
        self.selected_feats_ = self.all_feats_[self.get_support()]
        return self.transform(X), y


