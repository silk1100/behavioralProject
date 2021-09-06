import dill
import json
import os
import numpy as np
from collections import defaultdict


def createMLjson(ml_obj, output_dir):
    for rfe_key in ml_obj:
        ml_dict = ml_obj[rfe_key]
        for clc_name, clc in ml_dict.items():
            hyper_dict = clc.best_estimator_.__dict__
            modified_dict = {}
            tobedelted = (False,None)
            for key, val in hyper_dict.items():
                if key in ['_Booster', '_le', 'base_estimator_','estimators_','base_estimator',
                        '_tree','_y','_fit_X','loss_','init_','_rng']:
                    tobedeleted= (True, key)
                    print(tobedeleted)
                elif isinstance(val, np.ndarray):
                    modified_dict[key] = val.tolist()
                elif isinstance(val, np.int32):
                    modified_dict[key] = int(val)
                elif isinstance(val, np.int64):
                    modified_dict[key] = int(val)
                else:
                    modified_dict[key] = val
                    
            if tobedelted[0]:
                del hyper_dict[tobedeleted[1]]
            
            with open(os.path.join(output_dir, f'{rfe_key}_{clc_name}.json'), 'w') as f:
                try:
                    json.dump(modified_dict, f)
                except Exception:
                    print(modified_dict)

def createFSjson(FS_obj, output_dir):
    data_dict = defaultdict(dict)
    for rfe_key in FS_obj:
        fs_obj = FS_obj[rfe_key]
        fs_est = fs_obj.estimator_
        data_dict[rfe_key]['n_selected_feats'] = fs_est.n_features_in_
        data_dict[rfe_key]['n_input_feats'] = fs_obj.n_features_in_

        try:
            data_dict[rfe_key]['feat_imp'] = fs_est.feature_importances_.tolist()
        except:
            data_dict[rfe_key]['feat_imp'] = fs_est.coef_.tolist()

        data_dict[rfe_key]['grid_scores'] = fs_obj.grid_scores_.tolist()

    with open(os.path.join(output_dir, f'RFE.json'), 'w') as f:
        json.dump(data_dict, f)

        



if __name__ == "__main__":
    if not os.path.isdir('./ML_obj_hyperparams'):
        os.mkdir('./ML_obj_hyperparams')

    if not os.path.isdir('./RFE_obj_params'):
        os.mkdir('./RFE_obj_params')

    with open('./ML_obj.p', 'rb') as f:
        ml_obj = dill.load(f)
    
    with open('./FS_obj.p', 'rb') as f:
        fs_obj = dill.load(f)

    createFSjson(fs_obj, './RFE_obj_params')
    createMLjson(ml_obj, './ML_obj_hyperparams')
    
    
    