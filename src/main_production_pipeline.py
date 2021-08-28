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
# from utils import DataModelLoader, ProductionModelLoader
from ProductionModelCreator import ProductionModelCreator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (confusion_matrix, recall_score, f1_score, balanced_accuracy_score)
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
        """
        old code
        """
        # loader = DataModelLoader(self.models_dir)
        # self.models, self.json_data = loader.load()
        # loader = ProductionModelLoader('../selected_models_for_production/trained_normalizer_rfe_models.p')
        # self.models = loader.load()
        """
        END load code
        """
        loader = ProductionModelCreator(self.models_dir)
        loader.create_production_models_dict()
        self.models = loader.production_models_

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
            self.pipelines_dict_[behav_model] = (Pipeline(
                [
                    ('preprocessor', preprocessor),
                    ('ml', self.models[behav_model]['ml'])
                ]
            ), self.models[behav_model]['score'])
        return self.pipelines_dict_

    def _combine_pipelines(self, X):
        self.predictions_ = {}
        for name, (pipe, score) in self.pipelines_dict_.items():
            self.predictions_[name] = pipe.predict(X)
        return self.predictions_

    def adjust_predictions(self, predictions, score):
        if score <= 0.5:
            score = 0.5
        updated_predictions = np.array(predictions, dtype=np.double)
        for idx, pred in enumerate(predictions):
            if pred == 1:
                updated_predictions[idx] = pred + (1 - score)
            elif pred == 2:
                updated_predictions[idx] = pred - (1 - score)
        return updated_predictions

    def predict(self, X, method='performance_weighted'):
        self._create_pipelines() # needs to be in the __init__()
        self._combine_pipelines(X)
        predictions_list = []
        scores_list = []
        for key, predictions in self.predictions_.items():
            predictions_list.append(predictions)
            scores_list.append(self.pipelines_dict_[key][1])
        if method =='majority_voting':
            predictions_matrix = np.stack(predictions_list, axis=1)
            final_scores = stats.mode(predictions_matrix, axis=1)[0]
        elif method == 'weighted_average':
            predictions_matrix = np.zeros((len(predictions_list[0]), len(predictions_list)))
            for idx, pred in enumerate(predictions_list):
                adj_scores = self.adjust_predictions(pred, scores_list[idx])
                predictions_matrix[:, idx] = adj_scores
            final_scores = np.round(np.mean(predictions_matrix, axis=1))

        return final_scores



if __name__ == '__main__':
    b = BehavioralDiagnosis()
    # pipes = b._create_pipelines()
    # Prepare data to be used for testing
    # df = pd.read_csv('../notebooks/raw_data_for_production_testing.csv', index_col=0)
    df = pd.read_csv('../notebooks/raw_data_perc_for_production_testing.csv', index_col=0)
    
    srs_cols = [col for col in df.columns if 'SRS_' in col]
    df.dropna(inplace=True)
    original_labels = df['DX_GROUP'].values
    df.drop(['SEX', 'AGE_AT_SCAN ', 'DX_GROUP']+srs_cols, axis=1, inplace=True)
    my_label_cols = [col for col in df.columns if 'severity_label' in col]
    labels = df[my_label_cols]
    X = df.drop(my_label_cols, axis=1)
    y_hat = b.predict(X, 'weighted_average')
    C = confusion_matrix(original_labels, y_hat)
    tp = C[0, 0]
    tn = C[1, 1]
    fp = C[1, 0]
    fn = C[0, 1]
    print(f'sensitivity: {recall_score(y_true=original_labels, y_pred=y_hat)}')
    print(f'specificity : {tn / (tn + fp)}')
    print(f'balanced accuracy : {balanced_accuracy_score(y_true=original_labels, y_pred=y_hat)}')
    print(f'f1 : {f1_score(y_true=original_labels, y_pred=y_hat)}')
    print(f'confusion matrix : {C}')
    x=0