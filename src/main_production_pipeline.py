"""
1. A new subject(s) is/are given in form of a csv file with features extracted using FreeSurfer.
2. Each testing subject is going to pass through each behavioral ML model to get the diagnosis of that subject given
that model
3. Aggregation of all decisions to give the final classification for that subject
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.pipeline import Pipeline

import constants
from utils import DataModelLoader, ProductionModelLoader
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (confusion_matrix, recall_score, f1_score, accuracy_score, balanced_accuracy_score)
import dill

class NormalizeSelect(BaseEstimator, TransformerMixin):
    """
    Passed normalizer and rfe are expected to be trained and only used for transforming the data
    """
    def __init__(self, normalizer=None, rfe=None, **normalizer_rfe_params):
        self.normalizer = None
        self.rfe = None
        if normalizer:
            if isinstance(normalizer, str):
                with open(normalizer, 'rb') as f:
                    self.normalizer = dill.load(f)
            else:
                self.normalizer = normalizer
        if rfe:
            if isinstance(rfe, str):
                with open(rfe, 'rb') as f:
                    self.rfe = dill.load(f)
            else:
                self.rfe = rfe

        if self.normalizer is None:
            self._handle_norm_rfe_params(params_dict=normalizer_rfe_params)

    def _handle_norm_rfe_params(self, params_dict):
        args = list(params_dict.keys())
        if len(args) >= 1:
            for arg in args:
                if 'norm' in arg:
                    if isinstance(params_dict[arg], str):
                        with open(params_dict[arg], 'rb') as f:
                            self.normalizer = dill.load(f)
                    else:
                        self.normalizer = params_dict[arg]
                elif 'rfe' in arg:
                    if isinstance(params_dict[arg], str):
                        with open(params_dict[arg], 'rb') as f:
                            self.rfe = dill.load(f)
                    else:
                        self.rfe = params_dict[arg]


    def fit(self, X, y):
        return self

    def transform(self, X):
        Xs = self.normalizer.transform(X)
        Xselected = self.rfe.transform(Xs)
        return Xselected

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


class BehavioralDiagnosis:
    """
    Create a pipeline from all of the behavioral report models
    Feed to the pipeline the raw data of the testing subjects (extracted features from the FreeSUrfer)
    Get the final diagnosis
    """
    def __init__(self, method='performance_weighted', models_dir: str=None):
        self.method = method
        self.models_dir = models_dir if models_dir is not None else constants.MODELS_DIR['production']
        # loader = DataModelLoader(self.models_dir)
        # self.models, self.json_data = loader.load()
        loader = ProductionModelLoader('../selected_models_for_production/trained_normalizer_rfe_models.p')
        self.models = loader.load()
        print(f'Loaded behavioral models are: {[constants.SRS_TEST_NAMES_MAP[k] for k in self.models.keys()]}')

    # Obselete function. Currently there is a notebook "../notebooks/train_models_tobeusedin_production.ipynb" responsi-
    # ble for finding the best models (rfe, ml) and combine them together with their normalizer to be used directly in
    # production.
    # def _find_the_best_from_pseudo_metric_table(self, criteria='acc'):
    #     self._top_rfe_ml_per_model_ = {}
    #     for model in self.models:
    #         df_psudo_scores = pd.read_csv(os.path.join(self.models[model]['dir'],'pseudo_metrics.csv'), index_col=[0,1])
    #         max_loc = df_psudo_scores[criteria].argmax()
    #         rfe, ml = df_psudo_scores.index[max_loc]
    #         self._top_rfe_ml_per_model_[model] = (rfe, ml)
    #     return self._top_rfe_ml_per_model_

    def _create_pipelines(self):
        self.pipelines_dict_={}
        # self._find_the_best_from_pseudo_metric_table('acc')
        for behav_model in self.models:
            preprocessor = NormalizeSelect(normalizer=self.models[behav_model]['normalizer'],
                                           rfe=self.models[behav_model]['rfe'])
            self.pipelines_dict_[behav_model] = Pipeline(
                [
                    ('preprocessor', preprocessor),
                    ('ml', self.models[behav_model]['ml'])
                ]
            )
        return self.pipelines_dict_

    def _combine_pipelines(self, X):
        self.predictions_ = {}
        for name, pipe in self.pipelines_dict_.items():
            self.predictions_[name] = pipe.predict(X)
        return self.predictions_

    def predict(self, X, method='performance_weighted'):
        self._create_pipelines()
        self._combine_pipelines(X)
        predictions_list = []
        for _, predictions in self.predictions_.items():
            predictions_list.append(predictions)
        predictions_matrix = np.stack(predictions_list, axis=1)
        majority_voting = stats.mode(predictions_matrix, axis=1)[0]
        return majority_voting



if __name__ == '__main__':
    b = BehavioralDiagnosis()
    # pipes = b._create_pipelines()
    # Prepare data to be used for testing
    df = pd.read_csv('../notebooks/raw_data_for_production_testing.csv', index_col=0)
    srs_cols = [col for col in df.columns if 'SRS_' in col]
    df.dropna(inplace=True)
    original_labels = df['DX_GROUP'].values
    df.drop(['SEX', 'AGE_AT_SCAN ', 'DX_GROUP']+srs_cols, axis=1, inplace=True)
    my_label_cols = [col for col in df.columns if 'severity_label' in col]
    labels = df[my_label_cols]
    X = df.drop(my_label_cols, axis=1)
    y_hat = b.predict(X)
    C = confusion_matrix(original_labels, y_hat)
    tp = C[0, 0]
    tn = C[1, 1]
    fp = C[1, 0]
    fn = C[0, 1]
    print(f'senstivity: {recall_score(y_true=original_labels, y_pred=y_hat)}')
    print(f'specificity : {tn / (tn + fp)}')
    print(f'balanced accuracy : {balanced_accuracy_score(y_true=original_labels, y_pred=y_hat)}')
    print(f'f1 : {f1_score(y_true=original_labels, y_pred=y_hat)}')
    print(f'confusion matrix : {C}')
    x=0