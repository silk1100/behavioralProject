import constants
import os

exp_params = {
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
            'est': 'lsvm',#['xgb', 'lsvm', 'sgd','svm'],
            'cv':5,
            'scoring':'balanced_accuracy',
            'n_jobs':3,
            'verbose':3,
            'hyper_select_type':'random',
            'agg_models': False, #(need to be implemented)
            'n_iter':250
        }
    }





if __name__ == "__main__":
    if not os.path.isdir(constants.MODELS_DIR['config']):
        os.mkdir(constants.MODELS_DIR['config'])

