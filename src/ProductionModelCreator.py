import dill
import json
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.base import clone, BaseEstimator, TransformerMixin
import os
import utils 
import pandas as pd
import numpy as np



class RFEFeaturesBased(BaseEstimator, TransformerMixin):
    """
    Initialized by a directory of a folder that contains (selected_feats.json, group_df_beforeFixation.csv,
    pseudo_metrics.csv).
    Returns an object that can be used in a sklearn pipeline to select the features passed selected by the
    classifer in selected_feats.json which corresponds to the maximum accuracy score in psuedo_metrics.csv
    """
    def __init__(self, fldr):
        self.data_dir = fldr
        if not os.path.exists(os.path.join(fldr, 'selected_feats.json')):
            raise FileExistsError(f'There is no selected_feats.json inside {fldr}')
        self.selected_feats_json = os.path.join(fldr, 'selected_feats.json')
        
        if not os.path.exists(os.path.join(fldr, 'group_df_beforeFixation.csv')) and \
                not os.path.exists(os.path.join(fldr, 'group_df_afterFixation.csv')):
            raise FileExistsError(f'There is no group_df_beforeFixation.csv or group_df_afterFixation.csv inside {fldr}')
        if os.path.exists(os.path.join(fldr, 'group_df_afterFixation.csv')):
            self.df_dir = os.path.join(fldr, 'group_df_afterFixation.csv')
        else:
            self.df_dir = os.path.join(fldr, 'group_df_beforeFixation.csv')


        if not os.path.exists(os.path.join(fldr, 'pseudo_metrics.csv')):
            raise FileExistsError(f'There is no pseudo_metrics.csv inside {fldr}')
        self.metric_path = os.path.join(fldr, 'pseudo_metrics.csv')
        self.fit()

    def _load_features_from_json(self):
        with open(self.selected_feats_json, 'r') as f:
            feats_dict = json.load(f)
        return feats_dict

    def fit(self, X=None, y=None, **params):
        df = pd.read_csv(self.df_dir, index_col='subj_id')
        df_psudo_scores = pd.read_csv(self.metric_path, index_col=[0,1])
        max_loc = df_psudo_scores['acc'].argmax()
        rfe, ml = df_psudo_scores.index[max_loc]
        self.selected_rfe_name = rfe
        self.selected_ml_name = ml
        self.columns_ = df.columns
        self.feats_dict_ = self._load_features_from_json()
        self.feats_names_ = self.feats_dict_[rfe]
        self.feats_indices_ = []
        names_list = np.zeros(len(self.feats_dict_[rfe]), dtype=np.int32)
        for idx, feat_names in enumerate(self.feats_dict_[rfe]):
            names_list[idx] = self.columns_.get_loc(feat_names)
        self.feats_indices_ = names_list.copy()
        return self

    def transform(self, X, y=None, **params):
        if isinstance(X, pd.DataFrame):
            Xselected = X.loc[:, self.feats_names_]
        elif isinstance(X, np.ndarray):
            Xselected = X[:, self.feats_indices_]
        else:
            raise TypeError("I am only expecting dataframe or numpy.array")

        return Xselected




class ProductionModelCreator:
    """
    Upgrade over ProductionModelLoader:
    1. For the sake of full automation, I am adding whatever is done in "../notebooks/train_models_tobeusedin_production"
    to this class so that I can eliminate any source of bug by monitoring the whole process under the same code file
    2. The input should be the folder to all the interesting results (model/whateverexperimentyouwant)
    """
    def __init__(self, selected_models_dir):
        self.models_dir = selected_models_dir
        self._get_models_dir()
        self._get_top_models_for_each_folder()

    def _get_models_dir(self):
        self.fldrs_ = [os.path.join(self.models_dir, x) for x in os.listdir(self.models_dir)
                      if os.path.isdir(os.path.join(self.models_dir, x))]
        print(f'Result folders that are going to be used to create the production model: {self.fldrs_}')
    
    def _get_top_models_for_each_folder(self):
        self.top_rfe_ml_per_model_ = {}
        for fldr in self.fldrs_:
            df_psudo_scores = pd.read_csv(os.path.join(fldr,'pseudo_metrics.csv'), index_col=[0,1])
            max_loc = df_psudo_scores['acc'].argmax()
            rfe, ml = df_psudo_scores.index[max_loc]
            self.top_rfe_ml_per_model_[fldr] = (rfe, ml)
        print(f'Location of models: (RFE_OBJECT, ML_OBJECT)\n{self.top_rfe_ml_per_model_}')

    def _test_the_pipeline(self, fldr, normalizer, rfe_obj, ml_obj, behav_name):
        if os.path.isfile(os.path.join(fldr, 'group_df_afterFixation.csv')):
            df = pd.read_csv(os.path.join(fldr, 'group_df_afterFixation.csv'), index_col=0)
        else:
            df = pd.read_csv(os.path.join(fldr, 'group_df_beforeFixation.csv'), index_col=0)

        cols_2_del = ['DX_GROUP','AGE_AT_SCAN ', 'SEX']
        for col in df.columns:
            if 'categories_' in col:
                cols_2_del.append(col)
            elif 'SRS_' in col:
                cols_2_del.append(col)
        df.drop(cols_2_del, axis=1, inplace=True)

        X = df.drop('mylabels', axis=1)
        y = df['mylabels'].values
        Xs = normalizer.transform(X)
        Xselected = rfe_obj.transform(Xs)
        ml_obj_new = clone(ml_obj)
        trained_obj = ml_obj_new.fit(Xselected, y)
        yhat = trained_obj.predict(Xselected)
        score = balanced_accuracy_score(y, yhat)
        print(f'{behav_name}: {confusion_matrix(y,yhat)}')
        return trained_obj, score

    def create_production_models_dict(self):
        self.production_models_ = {}
        self.cross_platform_failures_ = []
        for fldr in self.fldrs_:
            behav_name = fldr.split('_')[-1]
            with open(os.path.join(fldr,'normalizer.p'), 'rb') as f:
                normalizer = dill.load(f)

            self.production_models_[behav_name] = {
                'normalizer': normalizer,
            }

            try:
                with open(os.path.join(fldr, 'FS_obj.p'), 'rb') as f:
                        fs_obj = dill.load(f)
            except:
                self.cross_platform_failures_.append(fldr)
                continue

            try:
                with open(os.path.join(fldr,'ML_obj.p'), 'rb') as f:
                    ml_obj = dill.load(f)
            except:
                self.cross_platform_failures_.append(fldr)
                continue       

            selected_rfe, selected_ml = self.top_rfe_ml_per_model_[fldr]
            
            rfe_obj = fs_obj[selected_rfe]
            ml_obj = ml_obj[selected_rfe][selected_ml].best_estimator_

            trained_obj, score = self._test_the_pipeline(fldr, normalizer=normalizer,
             rfe_obj=rfe_obj, ml_obj=ml_obj, behav_name=behav_name)

            self.production_models_[behav_name]['score'] = score
            self.production_models_[behav_name]['ml'] = trained_obj
            self.production_models_[behav_name]['rfe'] = fs_obj[selected_rfe]


        if len(self.cross_platform_failures_) > 0:
            print(f'fldrs that require cross platform fixation: {self.cross_platform_failures_}')
            print('Running _fix_cross_platform_issues() to fix the failed fldrs')
            self._fix_cross_platform_issues()

    def _fix_cross_platform_issues(self):
        for fldr in self.cross_platform_failures_:
            if not os.path.isdir(os.path.join(fldr, 'ML_obj_hyperparams')):
                raise FileNotFoundError(f'ML_obj_hyperparams folder hasnt been found in '
                f'doesnt exist in {fldr}')

            behav_name = fldr.split('_')[-1]
            rfe_obj = RFEFeaturesBased(fldr)
            clc_dict = utils.load_classifier_from_hyperparameterJson(os.path.join(fldr, 'ML_obj_hyperparams'))
            
            selected_ml = clc_dict[rfe_obj.selected_rfe_name][rfe_obj.selected_ml_name]
            trained_obj, score = self._test_the_pipeline(fldr,
             normalizer=self.production_models_[behav_name]['normalizer'],
            rfe_obj=rfe_obj,
            ml_obj=selected_ml,
            behav_name=behav_name)

            self.production_models_[behav_name]['score'] = score
            self.production_models_[behav_name]['ml'] = trained_obj
            self.production_models_[behav_name]['rfe'] = rfe_obj


if __name__ == '__main__':
    pmc = ProductionModelCreator('../tobedeletedmodelsfolder/')
    pmc.create_production_models_dict()