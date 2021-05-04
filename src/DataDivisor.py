"""
Classes that are responsible for splitting the data according to behavioral report, and block for confounding
variables such as age, gender, and IQ?

The logic behind the implementation of this class is 100% clinical.

Dr. Barnes Email.
"
The SRS T scores have a range: 59 or below is TD/normal. 60-65 is mild, 66-75 is moderate, and above 76 is severe.
Your data looks like that spread holds true.
"
"""
import os.path

import pandas as pd
import numpy as np
from constants import DATA_DIR, TARGET, ASD, TD
import constants


class DataDivisor:
    def __init__(self, data: pd.DataFrame, phenofile: pd.DataFrame, behavioral: str):
        self.df = data

        self.df_pheno = phenofile
        if 'subj_id' in self.df_pheno.columns:
            self.df_pheno.set_index('subj_id', inplace=True)

        self.report_type = behavioral

        self.behavioral_columns_ = []
        self._get_behavioral_columns()

    def _get_behavioral_columns(self):
        for col in self.df_pheno.columns:
            if self.report_type.lower() in col.lower():
                self.behavioral_columns_.append(col)

    def divide(self, phenotypes:dict=None, behavioral_type:str = None) -> pd.DataFrame:
        """
        phenotype should be a dictionary with keys as phenotype labels e.g (mild, moderate, sever).
        A new column will be added to the data with a label corresponding to each subject based on their
        behavioral_type score. If phenotypes is None, then the predefined phenotypes dict in constants.py is
        used. If behavioral_type is not given, then the total score will be used.
        :param phenotypes: dict
        :param behavioral_type: str
        :return: pd.DataFrame
        """
        def divisor(test_val):
            for key, val in constants.SRS_SCORES_MAP.items():
                if val[0]<=test_val<=val[1]:
                    return key
            return None

        # Currently we are only using the *_T score
        used_behav_rep = []
        for col in self.behavioral_columns_:
            if col.endswith('_T'):
                used_behav_rep.append(col)
        self.df_behav = self.df_pheno[used_behav_rep]
        updated_df = self.df.join(self.df_behav, how='inner')
        for srs_test in constants.SRS_TEST_T:
            df = updated_df.loc[:, self.df.columns.tolist()+[srs_test]]
            df[f'categories_{srs_test.split("_")[1]}'] = df[srs_test].apply(divisor)
            if not os.path.isdir(constants.DATA_DIV_DIR[srs_test]):
                os.mkdir(constants.DATA_DIV_DIR[srs_test])
            df.to_csv(os.path.join(constants.DATA_DIV_DIR[srs_test],f'{srs_test}.csv'),
                                   index_label='subj_id')

    def _validity_srs_test_type(self, srs_test_type:str) -> tuple:
        file_path = None
        correct_srs_test_type = srs_test_type
        if srs_test_type in constants.DATA_DIV_DIR:
            file_path = os.path.join(constants.DATA_DIV_DIR[srs_test_type], f'{srs_test_type}.csv')

        elif srs_test_type in constants.SRS_TEST_NAMES_MAP:
            file_path = os.path.join(
                constants.DATA_DIV_DIR[constants.SRS_TEST_NAMES_MAP[srs_test_type]],
                f'{constants.SRS_TEST_NAMES_MAP[srs_test_type]}.csv')
            correct_srs_test_type = constants.SRS_TEST_NAMES_MAP[srs_test_type]
        else:
            for srs_t in constants.SRS_TEST_NAMES_MAP[srs_test_type]:
                if srs_test_type in srs_t:
                    correct_srs_test_type = constants.SRS_TEST_NAMES_MAP[srs_test_type]
                    file_path = os.path.join(
                        constants.DATA_DIV_DIR[correct_srs_test_type],
                        f'{correct_srs_test_type}.csv')
                    break

        if file_path is None:
            raise ValueError(f'srs_test_type should be one of the following {list(constants.DATA_DIV_DIR.keys())}')

        return file_path, correct_srs_test_type

    def _validate_severity_level(self, severity_level:str) -> bool:
        if severity_level in constants.SEVERITY_LEVEL_AVAILABLE:
            return True
        return False

    def get_group(self, srs_test_type:str, severity_level:str, age_group:tuple=None, gender:str=None)->pd.DataFrame:
        try:
            file_path, correct_srs_test_type = self._validity_srs_test_type(srs_test_type)
        except Exception:
            raise ValueError(f'srs_test_type should be one of the following {list(constants.DATA_DIV_DIR.keys())}')

        if not self._validate_severity_level(severity_level):
            raise ValueError(f'severity level should be one of the following: {severity_level}')

        df =  pd.read_csv(file_path, index_col='subj_id')
        group_df = df[df[f'categories_{srs_test_type.split("_")[1]}'] == severity_level]

        return group_df

    def set_params(self, **params):
        pass
    
if __name__ == '__main__':
    df = pd.read_csv(DATA_DIR['medianMmedianP'], index_col=0)
    df_p = pd.read_csv(DATA_DIR['pheno'], index_col='subj_id')

    divisor = DataDivisor(df, df_p, 'srs')
    df = divisor.divide()


