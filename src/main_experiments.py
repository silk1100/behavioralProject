import os.path
import DataDivisor
import FeatureSelection
import Classifers
import constants
import dill
from experiment_designer import exp_params
import utils
import pandas as pd


class Experiment:
    def __init__(self, **experiment_params):
        self._expr_params = None
        self.stampfldr_ = None

        # Type of data to use
        self.data_repr = None

        # Data divisor parameters
        self.DD_srs_type = None
        self.DD_severity_group = None
        self.DD_age_group = None
        self.DD_gender = None
        self.DD_divide_data = False
        self._DD_obj = DataDivisor.DataDivisor()

        # Featureselection parameters
        self.FS_est = None
        self.FS_cv = None
        self.FS_scoring = None
        self.FS_n_jobs = None
        self.FS_verbose = None
        self.FS_step = None
        self.FS_min_features_to_select = None
        self._FS_obj = FeatureSelection.FeatureSelector()
        self.FS_selected_feats_ = None
        self.FS_grid_scores_ = None
        # Classifier Params
        self.ML_est = None
        self.ML_cv = None
        self.ML_scoring = None
        self.ML_n_jobs = None
        self.ML_verbose = None
        self.ML_hyper_search_type = None
        self.ML_agg_models = None # Only used when ML_set is a list of classifier keys
        self.ML_n_iter = None # Only used when hyper parameter select is randomized
        self._ML_obj = Classifers.CustomClassifier()
        self.ML_grid_ = None

        if len(experiment_params)> 1:
            self._parse_exp_params(experiment_params)
            self._expr_params = experiment_params

    def _check_and_fill_expr_params(self):
        for key, item in self.__dict__.items():
            parts = key.split('_')
            obj_type = parts[0]
            if obj_type in self._expr_params:
                self._expr_params[obj_type] = {'_'.join(parts[1:]):item}
            else:
                self._expr_params[obj_type]['_'.join(parts[1:])] = item

    def _check_and_fix_unbalance_groups(self, df_all:pd.DataFrame, df_group:pd.DataFrame)->pd.DataFrame:
        try:
            ASD_count = df_group['DX_GROUP'].value_counts()[1]
        except Exception:
            ASD_count = 0
        try:
            TD_count = df_group['DX_GROUP'].value_counts()[2]
        except Exception:
            TD_count = 0
        more_ASd = True
        if ASD_count > TD_count:
            ratio = TD_count/(ASD_count+TD_count)
        else:
            ratio = ASD_count/(TD_count+ASD_count)
            more_ASd = False

        if ratio < 0.4:
            # Sample within the age group of the more group
            if more_ASd:
                # Sample from TD
                df_td = df_all[df_all['DX_GROUP']==2]
                age_mean, age_std = df_group['AGE_AT_SCAN '].mean(), df_group['AGE_AT_SCAN '].std()
                upper_bound = age_mean+age_std
                lower_bound = age_mean-age_std
                df_target = df_td.loc[(df_td['AGE_AT_SCAN ']>=lower_bound) & (df_td['AGE_AT_SCAN ']<=upper_bound),:]
                df_added = df_target.sample(n=ASD_count-TD_count, random_state=1)
            else:
                # Sample from ASD
                df_asd = df_all[df_all['DX_GROUP']==1]
                age_mean, age_std = df_group['AGE_AT_SCAN '].mean(), df_group['AGE_AT_SCAN '].std()
                upper_bound = age_mean+age_std
                lower_bound = age_mean-age_std
                df_target = df_asd.loc[(df_asd['AGE_AT_SCAN ']>=lower_bound) & (df_asd['AGE_AT_SCAN ']<=upper_bound),:]
                df_added = df_target.sample(n=TD_count-ASD_count, random_state=1)
            df_added = df_added[df_group.columns]
            df = pd.concat([df_group, df_added], axis=0)
        else:
            df = df_group
        return df

    def run(self):
        stamp = utils.get_time_stamp()
        main_fldr = os.path.join(constants.MODELS_DIR['main'], stamp)
        self.stampfldr_ = main_fldr
        os.mkdir(main_fldr)

        utils.save_experiment_params(main_fldr, exp_params)

        if self._expr_params is None:
            self._check_and_fill_expr_params( )
        self._DD_obj.set_params(**self._expr_params['DD'])
        self._FS_obj.set_params(**self._expr_params['FS'])
        self._ML_obj.set_params(**self._expr_params['ML'])

        group_df = self._DD_obj.run()
        group_df.to_csv(os.path.join(main_fldr,'group_df_beforeFixation.csv'))

        # Make sure that the group_df contains TD and ASD
        group_df = self._check_and_fix_unbalance_groups(self._DD_obj._df_selected_groups_, group_df)
        group_df.to_csv(os.path.join(main_fldr,'group_df_afterFixation.csv'))

        # Drop Age, SEX, behavioral report, and behavioral category before feature selection
        age = group_df.pop('AGE_AT_SCAN ')
        sex = group_df.pop('SEX')
        _, srs_col_name = self._DD_obj._validity_srs_test_type(self.DD_srs_type)
        srs_col = group_df.pop(srs_col_name)
        srs_cat_col = group_df.pop(f'categories_{srs_col_name.split("_")[1]}')

        Xselected, y = self._FS_obj.run(group_df)
        utils.save_model(os.path.join(main_fldr, "FS_obj"), self._FS_obj)

        self.FS_selected_feats_ =  self._FS_obj.selected_feats_
        self.FS_grid_scores_ = self._FS_obj.scores_
        with open(os.path.join(main_fldr, 'selected_feats.txt'), 'w') as f:
            for feat in self.FS_selected_feats_:
                f.write(f'{feat}\n')
        with open(os.path.join(main_fldr, 'selected_feats_scores.txt'), 'w') as f:
            cntr = 0
            for score in self.FS_grid_scores_:
                f.write(f'{score} ')
                cntr += 1

        self.ML_grid_ = self._ML_obj.run(Xselected, y)
        utils.save_model(os.path.join(main_fldr, "ML_obj"), self._ML_obj)


    def _validate_params(self, dd:dict, prefix:str=None):
        for key, val in dd.items():
            if f'{prefix}_{key}' not in self.__dict__:
                raise KeyError(f'{prefix}_{key} is not a valid variable')

    def _parse_exp_params(self, exp_params_dict):
        for key, item in exp_params_dict.items():
            if key in ['DD', 'FS', 'ML']:
                data_dict = item
                self._validate_params(data_dict, key)
                for skey, sval in data_dict.items():
                    setattr(self, f'{key}_{skey}', sval)
            elif key == 'data_repr':
                self.data_repr = item
            else:
                raise KeyError(f'Experiment parameters should be one of the following ["DD","FS","ML"]')

    def set_params(self, **params):
        item = params.get(next(iter(params)))
        if isinstance(item, dict):
            self._parse_exp_params(params)

        for key, val in params.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
            else:
                raise ValueError(f'{key} is not a class member')

    def save_results(self, filename):
        # Needs to be modified
        if '.' in filename:
            postfix = filename.split('.')[1]
        else:
            postfix = 'p'

        with open(os.path.join(self.stampfldr_, filename), 'wb') as f:
            dill.dump(self, f)

        utils.save_experiment_params(self._expr_params)


    def visualize_results(self):
        pass

if __name__ == "__main__":
    experiment_1 = Experiment(**exp_params)
    experiment_1.run()
    experiment_1.save_results("singleFSML")

    # experiment_2 = Experiment(arguments2)
    # experiment_2.run()
    #
    # experiment_1.ML_est = "linear SVM"
    # params = {'FS_scoring':'balanced_accuracy',
    #                          'FS_n_jobs':-1,
    #                          'FS_verbose':3}
    # experiment_1.set_params(**params)
    #
    # print(experiment_1.__dict__)