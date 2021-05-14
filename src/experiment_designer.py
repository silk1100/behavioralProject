import constants
import os

exp_params = {
        'data_repr': 'median',
        'DD':{
            'srs_type': 'comm',
            'severity_group': 'sever',
            'age_group': None,
            'divide_data': True,
        },
        'FS':{
            'est': ['lsvm','lr'],
            'cv': 5,
            'scoring':'balanced_accuracy',
            'n_jobs':1,
            'verbose': 3,
            'step':1,
            'min_features_to_select': 1,
        },
        'ML':{
            'est': ['svm','nn','lsvm'],#['xgb', 'lsvm', 'sgd','svm'],
            'cv':5,
            'scoring':'balanced_accuracy',
            'n_jobs':3,
            'verbose':3,
            'hyper_search_type':'random',
            #'agg_models': False, #(need to be implemented)
            'n_iter':250
        }
    }





if __name__ == "__main__":
    if not os.path.isdir(constants.MODELS_DIR['config']):
        os.mkdir(constants.MODELS_DIR['config'])


