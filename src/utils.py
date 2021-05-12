import pandas as pd
import ujson as json
import datetime
import os
import dill

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
    if '.' in fname[-5:]:
        file = fname.split('.')[0]
    else:
        file = fname

    with open(file+'.p', 'wb') as f:
        dill.dump(model, f, recurse=True)


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