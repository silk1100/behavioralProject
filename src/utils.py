import pandas as pd
import ujson as json
import datetime
import os
import constants

def get_time_stamp():
    dt = datetime.datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")

def load_experiment(file):
    with open(file, 'r') as f:
        experiment = json.load(f)

    return experiment

def save_experiment_params(exp):
    json_files = [f for f in os.listdir(constants.MODELS_DIR['config']) if f.endswith('.json')]
    new_file = f'exp_{get_time_stamp()}.json'
    with open(os.path.join(constants.MODELS_DIR['config'], new_file), 'w') as f:
        json.dump(exp, f, indent=6)
    return new_file, len(json_files)

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