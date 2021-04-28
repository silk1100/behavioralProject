import pandas as pd


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