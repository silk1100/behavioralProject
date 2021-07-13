from main_production_pipeline import BehavioralDiagnosis
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score, precision_score

if __name__ == '__main__':
    # Get pipelines
    b = BehavioralDiagnosis()
    pipes = b._create_pipelines()

    # Load data ready for testing
    df = pd.read_csv('../notebooks/raw_data_for_production_testing.csv', index_col=0)

    # Get all sever subjects
    df_td = df[df['DX_GROUP']==2]
    df_asd = df[df['DX_GROUP'] == 1]
    df_td_extreme = df_td[
        (df_td['severity_label_comm']=='TD')&
        (df_td['severity_label_man']=='TD')&
        (df_td['severity_label_tot']=='TD')&
        (df_td['severity_label_mot']=='TD')&
        (df_td['severity_label_cog']=='TD')&
        (df_td['severity_label_awa']=='TD')
    ]
    df_asd_extreme = df_asd[
        (df_asd['severity_label_comm'] == 'sever')&
        (df_asd['severity_label_man'] == 'sever')&
        (df_asd['severity_label_tot'] == 'sever')&
        (df_asd['severity_label_mot'] == 'sever')&
        (df_asd['severity_label_cog'] == 'sever')&
        (df_asd['severity_label_awa'] == 'sever')
    ]
    df_extreme = pd.concat([df_td_extreme, df_asd_extreme], axis=0)

    # Preprocess the extreme data and feed it to the pipelines
    df_extreme = df_extreme.drop(['SEX', 'AGE_AT_SCAN ', 'DX_GROUP'], axis=1)
    df_extreme['mylabels'] = df_extreme['severity_label_comm'].apply(lambda x: 1 if x=='sever' else 2)
    df_extreme = df_extreme.drop(['severity_label_comm','severity_label_tot','severity_label_mot','severity_label_awa',
                                  'severity_label_cog','severity_label_man','SRS_TOTAL_T', 'SRS_AWARENESS_T',
                                  'SRS_COGNITION_T', 'SRS_COMMUNICATION_T', 'SRS_MOTIVATION_T', 'SRS_MANNERISMS_T'],
                                 axis=1)
    df_extreme.dropna(inplace=True)
    X = df_extreme.drop('mylabels', axis=1).values
    y = df_extreme['mylabels'].values
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=1234)

    # get predictions

    for test_name, pipe in pipes.items():
        pipe = pipe.fit(Xtrain, ytrain)
        y_hat = pipe.predict(Xtest)
        C = confusion_matrix(ytest, y_hat)
        tp = C[0, 0]
        tn = C[1, 1]
        fp = C[1, 0]
        fn = C[0, 1]
        print(f'senstivity of {test_name}: {recall_score(y_true=ytest, y_pred=y_hat)}')
        print(f'specificity of {test_name}: {tn / (tn + fp)}')
        print(f'accuracy of {test_name}: {accuracy_score(y_true=ytest, y_pred=y_hat)}')
        print(f'f1 of {test_name}: {f1_score(y_true=ytest, y_pred=y_hat)}')
        print(f'confusion matrix of {test_name}: {C}')

