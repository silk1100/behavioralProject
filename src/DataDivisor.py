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
        # self.df = self._handle_data_input(data)
        self.df = None
        self.df_pheno = self._handle_pheno_input(phenofile)
        if 'subjectkey' in self.df_pheno.columns:
            self.df_pheno.set_index('subjectkey', inplace=True)

        self.report_type = behavioral if behavioral is not None else 'srs'
        self.data_repr = 'medianMmedianP' if data is None else data
        self.srs_type=None
        self.severity_group=None
        self.age_group=None
        self.gender = None
        self.divide_data=False
        self.behavioral_columns_ = []
        self._df_selected_groups_ = None
        self.balance = False

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

        if 'subjectkey' in df.columns:
            df.set_index('subjectkey', inplace=True)
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
                file_path = os.path.join(constants.DATA_DIV_DIR[srs_test_type],
                                         f'{self.data_repr}_{srs_test_type}.csv')
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
                                 severity_level:tuple) -> bool:
        for severity in severity_level:
            if severity not in constants.SEVERITY_LEVEL_AVAILABLE:
                return False
        return True

    def get_group(self, srs_test_type:str, severity_level:tuple,
                  age_group:tuple=None, gender:str=None):
        try:
            file_path, correct_srs_test_type = self._validity_srs_test_type(srs_test_type)
        except Exception:
            raise ValueError(f'srs_test_type should be one of the following {list(constants.DATA_DIV_DIR.keys())}')

        if srs_test_type is None:
            return self.df

        if not self._validate_severity_level(severity_level):
            raise ValueError(f'severity level should be one of the following: {constants.SEVERITY_LEVEL_AVAILABLE}')

        idx0 = constants.SEVERITY_LEVEL_AVAILABLE.index(severity_level[0])
        idx1 = constants.SEVERITY_LEVEL_AVAILABLE.index(severity_level[1])
        if idx0 == idx1:
            raise ValueError("Cant run algorithm to classify same labels")

        idx_label = {}
        if idx0 < idx1:
            idx_label[severity_level[0]] = 2
            idx_label[severity_level[1]] = 1
        else:
            idx_label[severity_level[0]] = 1
            idx_label[severity_level[1]] = 2

        df = pd.read_csv(file_path, index_col='subj_id')
        self._df_selected_groups_ = df.copy()
        group_df = pd.DataFrame(columns=df.columns)
        for severity_group in severity_level:
            group = df[df[f'categories_{correct_srs_test_type.split("_")[1]}'] == severity_group]
            if age_group is not None:
                group = group[(age_group[0]<=group['AGE_AT_SCAN '])&(group['AGE_AT_SCAN ']<=age_group[1])]
                # Should I limit the age group over the whole data?? !
                df_ageLimited = df[(age_group[0]<=df['AGE_AT_SCAN '])&(df['AGE_AT_SCAN ']<=age_group[1])]
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

        group_df['mylabels'] = group_df[f'categories_{correct_srs_test_type.split("_")[1]}'].\
            apply(lambda x: idx_label[x])

        group_df.dropna(inplace=True)
        
        if self.balance:
            before_fixing = group_df.copy()
            group_df = self._check_and_fix_unbalance_groups(df if not age_group else df_ageLimited, group_df, idx_label )
            return before_fixing, group_df
        
        return group_df

    def set_params(self, **params) -> None:
        for key, val in params.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
            else:
                raise KeyError(f'{key} is not a valid parameter')

    def _check_and_fix_unbalance_groups(self, df_all:pd.DataFrame, df_group:pd.DataFrame, idx_label:dict,)->pd.DataFrame:
        def add_subjects2balance(df_all, df_group, added_group, converted_col_name, ASD_count, TD_count):
            if idx_label[added_group] == 2:
                severity_idx = constants.SEVERITY_LEVEL_AVAILABLE.index(added_group)
                if severity_idx == 0:
                    df_td_G = df_all[df_all['DX_GROUP'] == 2]  # _G stands for group
                else:
                    df_td_G = df_all[df_all[converted_col_name] == constants.SEVERITY_LEVEL_AVAILABLE[severity_idx-1]]  # _G stands for group

                age_mean, age_std = df_group['AGE_AT_SCAN '].mean(), df_group['AGE_AT_SCAN '].std()
                upper_bound = age_mean + age_std
                lower_bound = age_mean - age_std
                # df_target = df_td.loc[(df_td['AGE_AT_SCAN ']>=lower_bound) & (df_td['AGE_AT_SCAN ']<=upper_bound),:]
                df_target1 = df_td_G.loc[
                             (df_td_G['AGE_AT_SCAN '] >= lower_bound) & (df_td_G['AGE_AT_SCAN '] <= upper_bound), :]
                n_samples = constants.DD_MIN_N_PER_CLASS if len(df_td_G)>constants.DD_MIN_N_PER_CLASS else len(df_td_G)
                group_diff = ASD_count-TD_count
                if group_diff < n_samples:
                    df_added = df_target1.sample(group_diff, random_state=1)
                else:
                    df_added = df_target1.sample(n_samples, random_state=1)

                df_added['mylabels'] = 2
                df_added = df_added[df_group.columns]
                df_group = pd.concat([df_group, df_added], axis=0)
            elif idx_label[added_group] == 1:
                # Sample from ASD
                severity_idx = constants.SEVERITY_LEVEL_AVAILABLE.index(added_group)
                if severity_idx == len(constants.SEVERITY_LEVEL_AVAILABLE)-1:
                    df_asd = df_all[df_all['DX_GROUP'] == 1]  # _G stands for group
                else:
                    df_asd = df_all[df_all[converted_col_name] == constants.SEVERITY_LEVEL_AVAILABLE[severity_idx+1]]
                age_mean, age_std = df_group['AGE_AT_SCAN '].mean(), df_group['AGE_AT_SCAN '].std()
                upper_bound = age_mean + age_std
                lower_bound = age_mean - age_std
                df_target = df_asd.loc[
                            (df_asd['AGE_AT_SCAN '] >= lower_bound) & (df_asd['AGE_AT_SCAN '] <= upper_bound), :]
                n_samples = constants.DD_MIN_N_PER_CLASS if len(df_asd)>constants.DD_MIN_N_PER_CLASS else len(df_asd)
                group_diff = TD_count-ASD_count
                if group_diff < n_samples:
                    df_added = df_target.sample(group_diff, random_state=1)
                else:
                    df_added = df_target.sample(n_samples, random_state=1)

                df_added = df_added[df_group.columns]
                df_added['mylabels'] = 1
                df_group = pd.concat([df_group, df_added], axis=0)
            else:
                raise TypeError("Group should be either ASD or TD")
            return df_group

        def remove_subjects2balance(df_group, removed_group, ASD_count, TD_count):
            if removed_group == 'ASD':
                df_asd = df_group[df_group['mylabels']==1]
                df_toberemoved = df_asd.sample(n=ASD_count - TD_count, random_state=1234)
                updated_df = df_group.drop(df_toberemoved.index, axis=0)
            elif removed_group == 'TD':
                df_asd = df_group[df_group['mylabels']==2]
                df_toberemoved = df_asd.sample(n= TD_count - ASD_count, random_state=1234)
                updated_df = df_group.drop(df_toberemoved.index, axis=0)
            else:
                raise TypeError("Group should be either ASD or TD")
            return updated_df

        def get_key_corresponding_to_val(dict2search, value):
            for k, v in dict2search.items():
                if v == value:
                    return k
            return None

        category_col = [col for col in df_group.columns if col.startswith('categories_')][0].split('_')[1]
        converted_col_name = f"categories_{category_col}"
        try:
            ASD_count = df_group['mylabels'].value_counts()[1]
        except Exception:
            ASD_count = 0
        try:
            TD_count = df_group['mylabels'].value_counts()[2]
        except Exception:
            TD_count = 0
        more_ASd = True
        if ASD_count > TD_count:
            ratio = TD_count/(ASD_count+TD_count)
        else:
            ratio = ASD_count/(TD_count+ASD_count)
            more_ASd = False

        if ratio < 0.4:
            if more_ASd:
                cat_group = get_key_corresponding_to_val(idx_label, 2)
                if TD_count < constants.DD_MIN_N_PER_CLASS:
                    df_group = add_subjects2balance(df_all, df_group, cat_group, converted_col_name, ASD_count, TD_count)
                    try:
                        ASD_count = df_group['mylabels'].value_counts()[1]
                    except Exception:
                        ASD_count = 0
                    try:
                        TD_count = df_group['mylabels'].value_counts()[2]
                    except Exception:
                        TD_count = 0
                    if TD_count*1.0/(ASD_count+TD_count) < 0.4:
                        df = remove_subjects2balance(df_group, "ASD", ASD_count, TD_count)
                    else:
                        df = df_group
                else:
                    df = remove_subjects2balance(df_group, "ASD", ASD_count, TD_count)
            else:
                cat_group = get_key_corresponding_to_val(idx_label, 1)
                # Sample from ASD
                if ASD_count < constants.DD_MIN_N_PER_CLASS:
                    df_group = add_subjects2balance(df_all, df_group,cat_group, converted_col_name, ASD_count, TD_count)
                    try:
                        ASD_count = df_group['mylabels'].value_counts()[1]
                    except Exception:
                        ASD_count = 0
                    try:
                        TD_count = df_group['mylabels'].value_counts()[2]
                    except Exception:
                        TD_count = 0
                    if ASD_count*1.0/(ASD_count+TD_count) < 0.4:
                        df = remove_subjects2balance(df_group, "TD", ASD_count, TD_count)
                    else:
                        df = df_group
                else:
                    df = remove_subjects2balance(df_group, "TD", ASD_count, TD_count)

        else:
            df = df_group

        return df


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