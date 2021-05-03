import numpy as np
FREESURFER_FEATS_FILES = {
    'area': ['surf/lh.area', 'surf/rh.area'],
    'curv': ['surf/lh.curv', 'surf/rh.curv'],
    'thickness': ['surf/lh.thickness', 'surf/rh.thickness'],
    'volume': ['surf/lh.volume', 'surf/rh.volume']
}

FREESURFER_LABELS_FILES = {
    'a2009s': ['label/lh.aparc.a2009s.annot', 'label/rh.aparc.a2009s.annot'],
    'aparc': ['label/lh.aparc.annot', 'label/rh.aparc.annot']
}

ABIDEII_PATH = "/media/tarek/D/Autism/Structural_MRI/AbideII_FreeSurfer"
ABIDEI_PREPROC_PATH = "/media/tarek/D/Autism/Structural_MRI/Abide_Preprocessed_Dataset"

DATA_DIR = {
    'raw':'../data/raw',
    'raw_1file' : '../data/raw_allInOneJson',
    'feat_extract':'../data/feature_extraction',
    'data_divisor':'../data/data_divisor',
    'medianMmedianP':'../data/feature_extraction/medianMinusPlus/raw.csv',
    'percentile':'../data/feature_extraction/percentile/raw.csv',
    'pheno': '../data/updated_pheno.csv',
}

MODEL_DIR = {
    'main': '../models'
}


MAX_ITR = 1e9
PARAM_GRID={
    'lsmv': {
        'penalty':['l1','l2'],
        'loss':['hinge','squared_hinge'],
        'C':[0.1,1,5, 10]
    },
    'pagg': {
        'C':[0.1,1,5,10], 'n_iter_no_change':[1,5,10]
    },
    'sgd':{
        'loss':['modified_huber','squared_hinge','perceptron'],
        'penalty':['l1','l2','elasticnet'],
        'alpha': np.arange(0.1, 5, 0.1),
        'l1_ratio': np.arange(0,1,0.05),
    },
    'lr':{
        'penalty': ['l1','l2','elasticnet','none'],
        'C':[0.1,1,5, 10],
        'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    },
    'XGB': {
        'booster':['gbtree','gblinear','dart'],
        'learning_rate':[0.001, 0.01, 0.5, 1],
        'min_child_weight': [0.01, 0.5, 1, 10],
        'gamma': [0, 0.1, 1, 5, 50, 100],
        'reg_alpha':[0, 0.001, 0.5, 1, 10],
        'reg_lambda':[0, 0.001, 0.5, 1,  10],
        'colsample_bytree': [0.6, 0.8, 1.0],
    },
    'ridge':{
            'alpha':np.arange(0.1, 5, 0.1),
            'normalize':[True, False],
        },
    'knn': {
        'n_neighbors':np.arange(1,70, 2),
        'weights':['uniform','distance'],
        'leaf_size':np.arange(10,150, 10),
        'p':np.arange(1,7),
        'metric':['identifier','euclidean','manhattan','chebyshev','minkowski',
                  'wminkowski','seuclidean','mahalanobis']
        },
    'GNB': {

    },
    'rf': {
        'n_estimators':[50, 100, 200, 500, 1000],
        'criterion':['gini','entropy'],
        'max_features':['auto','sqrt'],
        'min_samples_split':[2,5,10],
        'min_samples_leaf':[0,0.1,0.2,0.3,0.4,0.5],
        'bootstrap':[True,False]

    },
    'svm': {
        'C':[0.1,1,5, 10],
        'kernel':['poly','rbf','sigmoid'],
        'degree':[2,3,4,5,6],
        'gamma':['scale','auto'],
        'coef0':[0.0,0.01,0.5,5,50,100]
    },
    'nn':{
        'hidden_layer_sizes': [(150,100,50,), (100,50,25,), (100,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001,0.001,0.01, 0.05, 0.1, 0.5],
        'learning_rate': ['constant','adaptive'],
        'beta_1':[0, 0.001, 0.01, 0.1, 0.3, 0.5, 0.9],
        'beta_2':[0, 0.001, 0.01, 0.1, 0.3, 0.5, 0.9],
    }
}

OUTPUT = {
    'lSVM': '../data/models/lSVM',
    'pagg': '../data/models/PAGG',
    'lg':'../data/models/LR',
    'XGB': '../data/models/XGB',
    'GNB': '../data/models/GNB',
    'Rf': '../data/models/RF',
    'SVC': '../data/models/SVC',
    'nn':'../data/models/nn'
}

SRS_SCORES_MAP = {
    'TD': (0, 59),
    'mild':(60, 65),
    'moderate':(66, 75),
    'sever':(76, 1000)
}

SRS_TEST_T = [
    'SRS_TOTAL_T',
    'SRS_AWARENESS_T',
    'SRS_COGNITION_T',
    'SRS_COMMUNICATION_T',
    'SRS_MOTIVATION_T',
    'SRS_MANNERISMS_T'
]
DATA_DIV_DIR = {
    'SRS_TOTAL_T': '../data/data_divisor/srs_total',
    'SRS_AWARENESS_T': '../data/data_divisor/srs_awar',
    'SRS_COGNITION_T': '../data/data_divisor/srs_cog',
    'SRS_COMMUNICATION_T': '../data/data_divisor/srs_comm',
    'SRS_MOTIVATION_T': '../data/data_divisor/srs_mot',
    'SRS_MANNERISMS_T': '../data/data_divisor/srs_manner',
}
SRS_TEST_NAMES_MAP = {
    'comm':'SRS_COMMUNICATION_T',
    'mot':'SRS_COMMUNICATION_T',
    'cog':'SRS_COGNITION_T',
    'awa':'SRS_AWARENESS_T',
    'man':'SRS_MANNERISMS_T',
    'tot':'SRS_TOTAL_T'
}

SEVERITY_LEVEL_AVAILABLE = [
    'TD','mild','moderate','sever'
]

TARGET = 'DX_GROUP'
ASD = 1
TD = 2