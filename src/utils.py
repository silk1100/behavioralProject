import pandas as pd
import ujson as json
import datetime
import os
import dill
from sklearn.pipeline import Pipeline
import constants
import json
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix)


def get_time_stamp():
    dt = datetime.datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")


def load_experiment(file):
    with open(file, 'r') as f:
        experiment = json.load(f)

    return experiment


def save_experiment_params(fldr, exp):
    new_file = f'exp_params.json'
    with open(os.path.join(fldr, new_file), 'w') as f:
        json.dump(exp, f, indent=6)


def save_model(fname, model):
    if fname.endswith('.p'):
        file = fname
    else:
        file = fname+'.p'

    with open(file, 'wb') as f:
        dill.dump(model, f)


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
        
        if not os.path.exists(os.path.join(fldr, 'group_df_beforeFixation.csv')):
            raise FileExistsError(f'There is no group_df_beforeFixation.csv inside {fldr}')
        self.df_dir = os.path.join(fldr, 'group_df_beforeFixation.csv')

        if not os.path.exists(os.path.join(fldr, 'pseudo_metrics.csv')):
            raise FileExistsError(f'There is no pseudo_metrics.csv inside {fldr}')
        self.metric_path = os.path.join(fldr, 'pseudo_metrics.csv')


    def _load_features_from_json(self):
        with open(self.selected_feats_json, 'r') as f:
            feats_dict = json.load(f)
        return feats_dict

    def fit(self, X=None, y=None, **params):
        df = pd.read_csv(self.df_dir, index_col='subj_id')
        df_psudo_scores = pd.read_csv(self.metric_path, index_col=[0,1])
        max_loc = df_psudo_scores['acc'].argmax()
        rfe, ml = df_psudo_scores.index[max_loc]
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


def create_hyperparameterJson_from_classifier(model_dir):
    output_dir = os.path.join(model_dir, "ML_obj_hyperparams")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(model_dir, 'ML_obj.p'), 'rb') as f:
        ml_obj = dill.load(f)

    for rfe_key in ml_obj:
        ml_dict = ml_obj[rfe_key]
        for clc_name, clc in ml_dict.items():
            hyper_dict = clc.best_estimator_.__dict__
            modified_dict = {}
            tobedelted = (False,None)
            for key, val in hyper_dict.items():
                if key in ['_Booster', '_le', 'base_estimator_','estimators_','base_estimator',
                        '_tree','_y','_fit_X','loss_','init_','_rng']:
                    tobedeleted= (True, key)
                    print(tobedeleted)
                elif isinstance(val, np.ndarray):
                    modified_dict[key] = val.tolist()
                elif isinstance(val, np.int32):
                    modified_dict[key] = int(val)
                elif isinstance(val, np.int64):
                    modified_dict[key] = int(val)
                else:
                    modified_dict[key] = val
                    
            if tobedelted[0]:
                del hyper_dict[tobedeleted[1]]
            
            with open(os.path.join(output_dir, f'{rfe_key}_{clc_name}.json'), 'w') as f:
                try:
                    json.dump(modified_dict, f)
                except Exception:
                    print(modified_dict)

def replace_nan_with_null_jsonfiles(jsonfilesdir, wordtoremove='NaN'):
    """
    jsonfilesdir: str directory with all json files require fixation
    """
    for jsonfile in [jsonfile for jsonfile in os.listdir(jsonfilesdir) if jsonfile.endswith('.json')]:
        with open(os.path.join(jsonfilesdir, jsonfile), 'r') as f:
            k = f.readline()
        if wordtoremove in k.lower():
            while wordtoremove in k.lower():
                idx = k.lower().index(wordtoremove)
                new_k = k[:idx]
                new_k += 'null'
                new_k += k[idx+len(wordtoremove):]
                k = new_k
            with open(os.path.join(jsonfilesdir, jsonfile), 'w') as f:
                f.write(k)
            print(f'Fixed {os.path.join(jsonfilesdir, jsonfile)}')
        

def load_classifier_from_hyperparameterJson(json_fldr):
    json_files = [x for x in os.listdir(json_fldr)]
    clc_dict = {}
    for file in json_files:
        rfe, ml = file.split('_')
        clc_dict[rfe] = {}
    for file in json_files:
        full_path = os.path.join(json_fldr, file)
        print(full_path)
        with open(full_path, 'r') as f:
            data = json.load(f)
        rfe_clc, ml_clc = file.split('_')
        clc = constants.CLC_DICT[ml_clc]()
        for x in clc.get_params().keys():
            if x not in data.keys():
                print(f'{file} missing {x}')
        hypparam_dict = {x:data[x] for x in clc.get_params().keys()}
        clc.set_params(**hypparam_dict)
        clc_dict[rfe_clc][ml_clc] = clc
    return clc_dict


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
        X = df.drop('my_labels', axis=1)
        y = df['my_labels'].values
        Xs = normalizer.transform(X)
        Xselected = rfe_obj.transform(Xs)
        trained_obj = ml_obj.fit(Xselected, y)
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

            try:
                with open(os.path.join(fldr,'ML_obj.p'), 'rb') as f:
                    ml_obj = dill.load(f)
            except:
                self.cross_platform_failures_.append(fldr)
                continue

            try:
                with open(os.path.join(fldr, 'FS_obj.p'), 'rb') as f:
                    fs_obj = dill.load(f)
            except:
                self.cross_platform_failures_.append(fldr)
                continue

            selected_rfe = self.top_rfe_ml_per_model_[fldr][0]
            selected_ml = self.top_rfe_ml_per_model_[fldr][1]
            rfe_obj = fs_obj[selected_rfe]
            ml_obj = ml_obj[selected_rfe][selected_ml].best_estimator_

            trained_obj, score = self._test_the_pipeline(fldr, normalizer=normalizer,
             rfe_obj=rfe_obj, ml_obj=ml_obj, behav_name=behav_name)

            self.production_models[behav_name] = {
                'normalizer': normalizer,
                'rfe': fs_obj[selected_rfe],
                'ml': trained_obj,
                'score': score
            }
        if len(self.cross_platform_failures_) > 0:
            print(f'fldrs that require cross platform fixation: {self.cross_platform_failures_}')
            print('Running _fix_cross_platform_issues() to fix the failed fldrs')
            self._fix_cross_platform_issues()

    def _fix_cross_platform_issues(self):
        for fldr in self.cross_platform_failures_:
            if not os.path.isdir(os.path.join(fldr, 'ML_obj_hyperparams')):
                raise FileNotFoundError(f'Will try to create json files out of the ML models because ML_obj_hyperparams folder'
                f'doesnt exist in {fldr}')

            clc_dict = load_classifier_from_hyperparameterJson(os.path.join(fldr, 'ML_obj_hyperparams'))
            





class ProductionModelLoader:
    """
    Load a dictionary containing an "almost" pipeline containing trained (Normalizer, RFE, trainedML) which are created
    by "../notebooks/train_models_tobeusedin_production".

    """
    def __init__(self, model_dict_fileloc:str=None):
        self.models_dict_dir = model_dict_fileloc if model_dict_fileloc is not None \
            else "../selected_models_for_production/trained_normalizer_rfe_models.p"

    def _check_if_model_dict_is_loaded(self):
        if 'models_dict_' in self.__dict__:
            return True
        return False

    def load(self):
        with open(self.models_dict_dir, 'rb') as f:
            self.models_dict_ = dill.load(f)
        return self.models_dict_

    def get_available_behavioral_tests(self):
        if not self._check_if_model_dict_is_loaded():
            raise BrokenPipeError('Run load() method before accessing any of the models')
        return list(self.models_dict_.keys()), [constants.SRS_TEST_NAMES_MAP[x] for x in self.models_dict_]

    def get_model_from_behavioral_test(self, behav_test, model_type):
        if not self._check_if_model_dict_is_loaded():
            raise BrokenPipeError('Run load() method before accessing any of the models')
        return self.models_dict_.get(behav_test).get(model_type)


class DataModelLoader:
    """
    Load a behavioral model to be used for prediction of a new subject(s)
    1. Need to load FS model
    2. ML best classifier
    3. Normalization method
    4. Create a pipeline to fuse all of those together and give out a prediction
    """
    def __init__(self, models_dir:str=None):
        if models_dir:
            self.models_dir = models_dir
        else:
            self.models_dir = constants.MODELS_DIR['production']

    def _load_norm_fs_ml(self, models_path):
        models_dict = {}
        for model in models_path:
            model_files = [os.path.join(model, x) for x in os.listdir(model) if x.endswith('.p')]
            module_name = model.split('\\')[-1].split('_')[-1]
            ml_model = None
            fs_model = None
            for file in model_files:
                obj_file_name = file.split('\\')[-1]
                if 'FS_' in obj_file_name:
                    try:
                        with open(file, 'rb') as f:
                            fs_model = dill.load(f)
                    except Exception:
                        fs_model = None
                elif 'ML_' in obj_file_name:
                    with open(file, 'rb') as f:
                        ml_model = dill.load(f)
                elif 'normalizer.' in obj_file_name:
                    with open(file, 'rb') as f:
                        normalizer = dill.load(f)
            models_dict[module_name] = {'dir':model,'normalizer':normalizer, 'fs': fs_model, 'ml':ml_model}

        return models_dict

    def _load_json(self, models_path):
        json_dict = {}
        for model in models_path:
            model_files = [os.path.join(model, x) for x in os.listdir(model) if x.endswith('.json')]
            module_name = model.split('\\')[-1].split('_')[-1]
            for file in model_files:
                obj_file_name = file.split('\\')[-1]
                if 'selected' in obj_file_name:
                    with open(file, 'r') as f:
                        selected_feats = json.load(f)
                elif 'exp' in obj_file_name:
                    with open(file, 'r') as f:
                        exp_des = json.load(f)

            json_dict[module_name] = {'exp': exp_des, 'sel': selected_feats}

        return json_dict

    def load(self):
        models_path = [os.path.join(self.models_dir, x) for x in os.listdir(self.models_dir)
                  if os.path.isdir(os.path.join(self.models_dir, x))]
        models_dict = self._load_norm_fs_ml(models_path)
        json_dict = self._load_json(models_path)
        return models_dict, json_dict


class DataFixation:
    def __init__(self):
        pass

    def _fix_val(self, key:str, val:str) -> str:
        if key == "KKI":
            return "_".join([val.split('_')[0], val.split('_')[-1]])
        if key == "U_MI":
            return "UM_" + val.split('_')[-1]
        if key == "SU_2":
            return "STANFORD_" + val.split('_')[-1]
        if key == "OILH":
            return "ONRC_2_part1_" + val.split('_')[-1]

    def remove_middle_1(self, df:pd.DataFrame, site_name: str) -> pd.DataFrame:
        indices = []
        vals = []
        for idx, val in df['subj_id'].items():
            if site_name in val:
                indices.append(idx)
                vals.append(self._fix_val(site_name, val))

        df.loc[indices, 'subj_id'] = vals
        return df




if __name__ == '__main__':
    # loader = DataModelLoader()
    # m_dict, j_dict = loader.load()
    # x=0
    clc_dict = load_classifier_from_hyperparameterJson('../models/20210815_170500_perc_ubuntu_mot/ML_obj_hyperparams')
    print(clc_dict)