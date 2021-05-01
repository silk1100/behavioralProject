from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

from sklearn import base

class FeatureSelector(base.TransformerMixin, base.BaseEstimator):
    def __int__(self, est, dataDivisor_dir, behavioral_div, selector_method='rfe'):
        self.dir = dataDivisor_dir
        self.behavior = behavioral_div
        self.selected_feats_ = None
        self.scores_ = None
        self.est = est
        self.default_rfecv = {
            'cv':StratifiedKFold(5, shuffle=True, random_state=123),
            'scoring': 'accuracy',
            'n_jobs': -1,
            'verbose': 3,
            'step':1,
            'min_features_to_select':1
        }
        self.rfe_=None

    def fit(self, X, y, **rfecv_params):
        data_dict = {**self.default_rfecv, **rfecv_params} if rfecv_params is not None else self.default_rfecv
        self.rfe_ = RFECV(self.est,**data_dict)
        self.rfe_.fit(X, y)
        return self

    def transform(self, X):
        if self.rfe_ is None:
            raise ValueError("You need to call fit function first")
        return self.rfe_.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        self.rfe_.fit(X, y)
        return self.rfe_.transform(X)
    