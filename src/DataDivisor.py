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
import os
import pandas as pd
import numpy as np
from constants import DATA_DIR, TARGET, ASD, TD
import constants


class DataDivisor:
    def __init__(self, data: pd.DataFrame=None, phenofile: pd.DataFrame=None,
                 behavioral: str=None):
        self.df = self._handle_data_input(data)

        self.df_pheno = self._handle_pheno_input(phenofile)
        if 'subj_id' in self.df_pheno.columns:
            self.df_pheno.set_index('subj_id', inplace=True)

        self.report_type = behavioral if behavioral is not None else 'srs'
        self.data_repr = 'medianMmedianP' if data is None else data
        self.srs_type=None
        self.severity_group=None
        self.age_group=None
        self.gender = None
        self.divide_data=False
        self.behavioral_columns_ = []
        self._df_selected_groups_ = None

        self._get_behavioral_columns()

    def _handle_pheno_input(self, pheno):
        """
        Handling pheno as a parameter passed to the constructor
        :return:
        """
        if pheno is None:
            df = pd.read_csv(constants.DATA_DIR['pheno'])
        elif isinstance(pheno, pd.DataFrame):
                df = pheno
        else:
            raise TypeError(f'pheno can be either None or dataframe')

        if 'subj_id' in df.columns:
            df.set_index('subj_id', inplace=True)
            if 'Unnamed: 0' in df.columns:
                df.drop('Unnamed: 0', axis=1, inplace=True)
        else:
            df.set_index('Unnamed: 0', inplace=True)

        return df

    def _handle_data_input(self, data):
        """
        Handling data as a parameter passed to the constructor
        :return:
        """
        if data is None:
            print('medianMmedianP is used as a default since no data is passed')
            df = pd.read_csv(constants.DATA_DIR['medianMmedianP'])
        else:
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, str):
                if 'median' in data:
                    file_path = constants.DATA_DIR['medianMmedianP']
                elif 'percentile' in data:
                    file_path = constants.DATA_DIR['percentile']
                else:
                    raise ValueError(f'string represtation of data should be one of the following: {self.FEATURE_REPR}')
                df = pd.read_csv(file_path)
            else:
                raise TypeError(f'{data} is expected to be the feature data frame, str to the type of features'
                                f'{constants.DATA_REPR_MAP.keys()}')
        if 'subj_id' in df.columns:
            df.set_index('subj_id', inplace=True)
        else:
            df.set_index('Unnamed: 0', inplace=True)

        # Remove columns of unknown brain regions
        cols2remove = [col for col in df.columns if ('lunknown' in col)or('runknown' in col)]
        df.drop(cols2remove, axis=1, inplace=True)

        return df

    def _get_behavioral_columns(self):
        for col in self.df_pheno.columns:
            if self.report_type.lower() in col.lower():
                self.behavioral_columns_.append(col)

    def divide(self, phenotypes:dict=None,
               behavioral_type:str=None) -> pd.DataFrame:
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

        # Currently we are only using the *_T score (total)
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
            df.to_csv(os.path.join(constants.DATA_DIV_DIR[srs_test],f'{self.data_repr}_{srs_test}.csv'),
                                   index_label='subj_id')

    def _validity_srs_test_type(self,
                                srs_test_type:str) -> tuple:
        if srs_test_type is None:
            return None, None
        file_path = None
        correct_srs_test_type = srs_test_type
        if srs_test_type in constants.DATA_DIV_DIR:
            file_path = os.path.join(constants.DATA_DIV_DIR[srs_test_type], f'{self.data_repr}_{srs_test_type}.csv')

        elif srs_test_type in constants.SRS_TEST_NAMES_MAP:
            file_path = os.path.join(
                constants.DATA_DIV_DIR[constants.SRS_TEST_NAMES_MAP[srs_test_type]],
                f'{self.data_repr}_{constants.SRS_TEST_NAMES_MAP[srs_test_type]}.csv')
            correct_srs_test_type = constants.SRS_TEST_NAMES_MAP[srs_test_type]
        else:
            for srs_t in constants.SRS_TEST_NAMES_MAP[srs_test_type]:
                if srs_test_type in srs_t:
                    correct_srs_test_type = constants.SRS_TEST_NAMES_MAP[srs_test_type]
                    file_path = os.path.join(
                        constants.DATA_DIV_DIR[correct_srs_test_type],
                        f'{self.data_repr}_{correct_srs_test_type}.csv')
                    break

        if file_path is None:
            raise ValueError(f'srs_test_type should be one of the following {list(constants.DATA_DIV_DIR.keys())}')

        return file_path, correct_srs_test_type

    def _validate_severity_level(self,
                                 severity_level:str) -> bool:
        for severity in severity_level:
            if severity not in constants.SEVERITY_LEVEL_AVAILABLE:
                return False
        return True

    def get_group(self, srs_test_type:str, severity_level:str,
                  age_group:tuple=None, gender:str=None)->pd.DataFrame:
        try:
            file_path, correct_srs_test_type = self._validity_srs_test_type(srs_test_type)
        except Exception:
            raise ValueError(f'srs_test_type should be one of the following {list(constants.DATA_DIV_DIR.keys())}')

        if srs_test_type is None:
            return self.df

        if not self._validate_severity_level(severity_level):
            raise ValueError(f'severity level should be one of the following: {constants.SEVERITY_LEVEL_AVAILABLE}')

        df = pd.read_csv(file_path, index_col='subj_id')
        self._df_selected_groups_ = df
        group_df = pd.DataFrame(columns=df.columns)
        for severity_group in severity_level:
            group = df[df[f'categories_{correct_srs_test_type.split("_")[1]}'] == severity_group]
            if age_group is not None:
                group = group[age_group[0]<=group['AGE_AT_SCAN']<=age_group[1]]
            if gender is not None:
                if gender in 'male' or gender.lower() == 'm' or gender == 1:
                    group = group[group['SEX']==1]
                elif gender in 'female' or gender.lower() == 'f' or gender == 2:
                    group = group[group['SEX'] == 2]
                else:
                    raise ValueError(f'Gender can be either (male/m/1) or (female/f/2)')
            if len(group_df)==0:
                group_df = group
            else:
                group_df = pd.concat([group_df, group], axis=0)

        return group_df

    def set_params(self, **params) -> None:
        for key, val in params.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
            else:
                raise KeyError(f'{key} is not a valid parameter')

    def run(self):
        # self.srs_type=None
        # self.severity_group=None
        # self.age_group=None
        # self.gender = None
        # self.divide_data=False
        if self.divide_data:
            self.divide()
        return self.get_group(self.srs_type, self.severity_group, self.age_group, self.gender)


if __name__ == '__main__':
    df = pd.read_csv(DATA_DIR['medianMmedianP'], index_col=0)
    df_p = pd.read_csv(DATA_DIR['pheno'], index_col='subj_id')

    # divisor = DataDivisor(df, df_p, 'srs')
    # df = divisor.divide()
    divisor = DataDivisor()
    df = divisor.get_group('comm','sever')
    x = 0