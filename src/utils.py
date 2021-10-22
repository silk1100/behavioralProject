from numpy import matrixlib
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
from sklearn.base import clone
import warnings
warnings.filterwarnings('ignore')


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
        if file.endswith('.json'):
            rfe_clc, ml_clc = file.split('.')[0].split('_')
        else:
            rfe_clc, ml_clc = file.split('_')

        clc = constants.CLC_DICT[ml_clc]()
        for x in clc.get_params().keys():
            if x not in data.keys():
                print(f'{file} missing {x}')
        hypparam_dict = {x:data[x] for x in clc.get_params().keys()}
        clc.set_params(**hypparam_dict)
        clc_dict[rfe_clc][ml_clc] = clc
    return clc_dict




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
    pass