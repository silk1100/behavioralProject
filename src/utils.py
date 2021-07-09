import pandas as pd
import ujson as json
import datetime
import os
import dill
from sklearn.pipeline import Pipeline
import constants

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

    def _load_fs_ml(self, models_path):
        models_dict = {}
        for model in models_path:
            model_files = [os.path.join(model, x) for x in os.listdir(model) if x.endswith('.p')]
            module_name = model.split('\\')[-1].split('_')[-1]
            ml_model = None
            fs_model = None
            for file in model_files:
                obj_file_name = file.split('\\')[-1]
                if 'FS' in obj_file_name:
                    try:
                        with open(file, 'rb') as f:
                            fs_model = dill.load(f)
                    except Exception:
                        fs_model = None
                elif 'ML' in obj_file_name:
                    with open(file, 'rb') as f:
                        ml_model = dill.load(f)
            models_dict[module_name] = {'fs': fs_model, 'ml':ml_model}

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
        models_dict = self._load_fs_ml(models_path)
        json_dict = self._load_fs_ml(models_path)
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
    loader = DataModelLoader()
    m_dict, j_dict = loader.load()
    x=0