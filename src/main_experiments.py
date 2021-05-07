import DataDivisor
import FeatureSelection
import Classifers
import constants
import dill
import ujson as json

class Experiment:
    def __init__(self, **experiment_params):
        if len(experiment_params)> 1:
            self._parse_exp_params(experiment_params)
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
        self.FS_min_feat_to_select = None
        self._FS_obj = FeatureSelection.FeatureSelector()

        # Classifier Params
        self.ML_est = None
        self.ML_cv = None
        self.ML_scoring = None
        self.ML_n_jobs = None
        self.ML_verbose = None
        self.ML_hyper_select_type = None
        self.ML_agg_models = None # Only used when ML_set is a list of classifier keys
        self.ML_n_iter = None # Only used when hyper parameter select is randomized
        self._ML_obj = Classifers.CustomClassifier()

        print("I am constructor")

    def run(self):
        DD_params_dict = {}
        FS_params_dict = {}
        ML_params_dict = {}
        for key, val in self.__dict__.items():
            if key.startswith('DD'):


        self._DD_obj.set_params()

    def _parse_exp_params(self, exp_params_dict):
        for key, item in exp_params_dict:
            if 'DD' == key:
                DD_data_dict = item
            elif 'FS' == key:
                FS_data_dict = item
            elif 'ML' == key:
                ML_data_dict = item
            else:
                raise KeyError(f'Experiment parameters should be one of the following ["DD","FS","ML"]')

    def set_params(self, **params):
        for key, val in params.items():
            if key in self.__dict__.keys():
                setattr(self, key, val)
            else:
                raise ValueError(f'{key} is not a class member')

    def save_results(self, filename):
        if '.' not in filename:
            filename += '.p'

        with open(filename, 'rb') as f:
            dill.dump(self, f)

        # Until I make sure that dill can load my Experiment class anywhere
        new_file_name = filename.split('.')[0]
        new_file_name += '_sklearnGrid.p'
        with open(new_file_name, 'rb') as f:
            dill.dump(self.ML_g, f)


    def visualize_results(self):
        pass

if __name__ == "__main__":
    params_experiment_1 = {
        'DD':{
            'srs_type': 'comm',
            'severity_group': 'severity',
            'age_group': None,
            'divide_data': False,
        },
        'FS':{
            'est': 'lsvm',
            'cv': 5,
            'scoring':'balanced_accuracy',
            'n_jobs':3,
            'verbose': 3,
            'step':1,
            'min_feat_to_select': 1,
        },
        'ML':{
            'est':['xgb', 'lsvm', 'sgd','svm'],
            'cv':5,
            'scoring':'balanced_accuracy',
            'n_jobs':3,
            'verbose':3,
            'hyper_select_type':'random',
            'agg_models': False #(need to be implemented)
            'n_iter':250
        }
    }
    experiment_1 = Experiment()
    experiment_1.run()
    experiment_1.save_results()
    dill.dump(recurse=True)


    experiment_2 = Experiment(arguments2)
    experiment_2.run()

    experiment_1.ML_est = "linear SVM"
    params = {'FS_scoring':'balanced_accuracy',
                             'FS_n_jobs':-1,
                             'FS_verbose':3}
    experiment_1.set_params(**params)

    print(experiment_1.__dict__)