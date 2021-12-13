import json
import os
from collections import defaultdict

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, BaggingClassifier, VotingClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (classification_report, auc, roc_auc_score, roc_curve, f1_score,
                             recall_score, precision_score, balanced_accuracy_score, confusion_matrix)

import pandas as pd
import numpy as np
from Classifers import CustomClassifier
import constants
from ProductionModelCreator import ProductionModelCreator
import dill
from scipy import stats


class TestSubjectsLoader:
    def __init__(self):
        pass


class Phase2:
    def __init__(self, best_scores_dict: str, custom_clc_dict: dict):
        self.best_score_dict = self._read_dict(best_scores_dict)
        self.models_ = defaultdict(dict)
        self.df = pd.read_csv(constants.DATA_DIR['phase2'], index_col=0)
        self.df.dropna(inplace=True)
        self.y = self.df['DX_GROUP']
        cols = [col for col in self.df.columns if '_label_' in col or 'SRS' in col]
        self.X = self.df.drop(['DX_GROUP', 'AGE_AT_SCAN ', 'SEX']+cols, axis=1)
        self.clc = CustomClassifier()
        self.clc.set_params(**custom_clc_dict)
        self.X_feats = None

    def _read_dict(self, best_score_dir):
        with open(best_score_dir, 'r') as f:
            d = json.load(f)
        return d

    def _load_models(self):
        for sev, sev_dict in self.best_score_dict.items():
            for beh, beh_dict in sev_dict.items():
                exp_dir = beh_dict['exp']
                feats = beh_dict['feats']
                data_dir = os.path.join(constants.OUTPUT_DIR_CLUSTER, exp_dir, sev,
                             f"AgebetweenNonetNone_{sev}_percentile_minmax",f"percentile_minmax_{sev}_{beh}")
                normalizer_dir = os.path.join(data_dir, "normalizer.p")
                ml_dir = os.path.join(data_dir, "ML_obj.p")
                with open(normalizer_dir, 'rb') as f:
                    normalizer = dill.load(f)
                with open(ml_dir, "rb") as f:
                    ml = dill.load(f)
                self.models_[sev][beh] = {
                    'normalizer': normalizer,
                    'feats': feats,
                    'clc': ml[beh_dict['rfe']][beh_dict['clc']].best_estimator_
                }

    def _sigmoid(self, v):
        return 1/(1+np.exp(-v))

    def _create_feature_matrix(self):
        X_prob2beASD = pd.DataFrame(index=self.X.index)
        for sev, sev_dict in self.models_.items():
            for beh, beh_dict in sev_dict.items():
                feat_idx = [self.X.columns.get_loc(feat) for feat in beh_dict['feats']]
                Xnorm = beh_dict['normalizer'].transform(self.X)
                Xselected = Xnorm[:, feat_idx]
                if 'LinearSVC' in str(beh_dict['clc']):
                    prob = 1-self._sigmoid(beh_dict['clc'].decision_function(Xselected))
                    X_prob2beASD[f'{sev}_{beh}'] = prob
                else:
                    prob = beh_dict['clc'].predict_proba(Xselected)
                    X_prob2beASD[f'{sev}_{beh}'] = prob[:, np.where(beh_dict['clc'].classes_== 1)[0].tolist()[0]]

        return X_prob2beASD

    def score(self):
        self._load_models()
        self.X_feats = self._create_feature_matrix()
        Xtrainvalid, Xtest, ytrainvalid, ytest = train_test_split(self.X_feats, self.y, test_size=0.2, shuffle=True,
                                                                  random_state=123)
        self.ML_grid = self.clc.run(Xtrainvalid, ytrainvalid, est="None")
        best_score = 0
        best_estimator = None
        best_clc = None
        for clc, grid in self.ML_grid['None'].items():
            if grid.best_score_ > best_score:
                best_score = grid.best_score_
                best_estimator = grid.best_estimator_
                best_clc = clc

        print(f"Highest score is {best_score} for {best_clc}")
        yhat = best_estimator.predict(Xtest)
        print(f"f1 score: {f1_score(ytest, yhat)}")
        print(f"percesion score: {precision_score(ytest, yhat)}")
        print(f"recall score: {recall_score(ytest, yhat)}")
        print(f"balanced acc score: {balanced_accuracy_score(ytest, yhat)}")
        print(f"confusion matrix: {confusion_matrix(ytest, yhat)}")

    def fit(self, X, y):
        pass


def main():
    cclc_dict = {
        "est": [
            'lr',
            'xgb',
            'rf',
            'nn'
            # "svm",
            # "lgbm",
            # "nn"
        ],
        "cv": 5,
        "scoring": "balanced_accuracy",
        "n_jobs": -1,
        "verbose": 0,
        "hyper_search_type": "random",
        "n_iter": 15
    }
    p2 = Phase2(constants.BEST_RESULTS_JSON, cclc_dict)
    p2.score()


if __name__ == "__main__":
    main()
