import constants
import os

exp_params = {
        'data_repr': 'median',
        'normalizer':'std',
        'DD':{
            'srs_type': 'comm',#'cog',comm
            'severity_group': 'sever',
            'age_group': None,
            'divide_data': False,
        },
        'FS':{
            # 'est': ['lsvm','lr','xgb'], # Either this or the directory to a model folder with ML_obj.p in it (as below)
            'est': "../models/FS_Hyperparameters_Comm/", # If it is a directory, then Read the classifiers in MLobj
            'cv': 5,
            'scoring':'balanced_accuracy',
            'n_jobs':-1,
            'verbose': 3,
            'step':1,
            'min_features_to_select': 1,
        },
        'ML':{
            'est': ['lr','lsvm','xgb','rf', 'gbt','knn','gnb','gbt', 'svm'],#['svm','nn','lsvm','xgb'],#['xgb', 'lsvm', 'sgd','svm'],
            'cv':5,
            'scoring':'balanced_accuracy',
            'n_jobs':-1,
            'verbose':3,
            'hyper_search_type':'random',
            #'agg_models': False, #(need to be implemented)
            'n_iter':500
        }
    }





if __name__ == "__main__":
    if not os.path.isdir(constants.MODELS_DIR['config']):
        os.mkdir(constants.MODELS_DIR['config'])


