from multiprocessing.sharedctypes import Value
import numpy as np
from probabilityOfSuccessComputations import create_prob
import pandas as pd
import dill
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import seaborn as sns
from math import comb
sns.set()


"""
CONSTANTS
"""
MAIN_DIR = "../output/bigdatacluster/randomizedOutput"
OUTPUT_DIR = "./.neuroatlas_results"
experiment_folders = [x for x in os.listdir(MAIN_DIR) if os.path.isdir(os.path.join(MAIN_DIR, x))]
df_sample = pd.read_csv("../output/bigdatacluster/randomizedOutput/experiment_adj2/sever_TD/AgebetweenNonetNone_sever_TD_percentile_minmax/percentile_minmax_sever_TD_awa/"
                        "group_df_beforeFixation.csv", index_col=0)
ALLFEATS = [x for x in df_sample.columns if '_PERC' in x]

"""
Create the main dataframe which will contain all the results
"""
def create_exp_result_df(expr_fldrs):
    data_dict = defaultdict(list)
    for experiment in expr_fldrs:
        full_dir = os.path.join(MAIN_DIR, experiment)
        for severity_dir in [x for x in os.listdir(full_dir) if os.path.isdir(os.path.join(full_dir, x))]:
            full_seve_dir = os.path.join(full_dir, severity_dir, f"AgebetweenNonetNone_{severity_dir}_percentile_minmax")
            for beh_dir in [x for x in os.listdir(full_seve_dir) if os.path.isdir(os.path.join(full_seve_dir, x))]:
                full_beh_dir = os.path.join(full_seve_dir, beh_dir)

                df_pseudo = pd.read_csv(os.path.join(full_beh_dir, "pseudo_metrics.csv"))
                df_pseudo['avg'] = df_pseudo[['acc','f1']].mean(axis=1)
                best_rfe, best_clc = df_pseudo[['RFE','Metrics']].iloc[df_pseudo['avg'].argmax()]

                with open(os.path.join(full_beh_dir, "selected_feats.json"), 'r') as f:
                    rfe_dict = json.load(f)
                feats = rfe_dict[best_rfe]

                with open(os.path.join(full_beh_dir, "ML_obj.p"), 'rb') as f:
                    ML_obj = dill.load(f)

                clc = ML_obj[best_rfe][best_clc]
                data_dict['exp'].append(experiment)
                data_dict['sev'].append(severity_dir)
                data_dict['beh'].append(beh_dir.split('_')[-1])
                data_dict['rfe'].append(best_rfe)
                data_dict['clc'].append(best_clc)
                data_dict['feats'].append(feats)
                data_dict['hyper_params'].append(clc.best_params_)
                data_dict['clc_score'].append(clc.best_score_)
                data_dict['pseudo_acc'].append(df_pseudo['acc'].iloc[df_pseudo['avg'].argmax()])
                data_dict['pseudo_f1'].append(df_pseudo['f1'].iloc[df_pseudo['avg'].argmax()])

                df_results = pd.DataFrame(data_dict)
    return df_results


##########################################################################################################################################################
"""
STATISTICS FUNCTIONS
"""
##########################################################################################################################################################
def create_best_data_dict(df):
    def get_best_severity_behavioral(df, sev, beh):
        sub_df = df[(df['sev']==sev)&(df['beh']==beh)]
        sub_df['tot'] =  (sub_df['clc_score']+ sub_df['pseudo_acc'] + sub_df['pseudo_f1'])/3 - len(sub_df['feats'])/1088
        return sub_df.iloc[sub_df['tot'].argmax()]

    best_data_dict = defaultdict(dict)
    for sev in ['mild_TD', 'moderate_TD', 'sever_TD']:
        for beh in ['awa','comm','man','cog','mot', 'tot']:
            best_data_dict[sev][beh] = get_best_severity_behavioral(df, sev, beh).to_dict()

    with open(os.path.join(OUTPUT_DIR, 'best_results_dict.json'), 'w') as f:
        json.dump(best_data_dict, f)

    return best_data_dict


def _parse_feats_as_list(featsList_str):
    featsList_str = featsList_str[1:-1]
    feats_list = featsList_str.split(',')
    output = []
    for feat in feats_list:
        feat = feat.replace("'", "")
        output.append(feat)
    
    return output

def _neuroAtlas_allSev_allBeh(best_data_dict):
    neuroAtlas_agg_dict = {}
    for sev, sev_dict in best_data_dict.items():
        for beh, beh_dict in sev_dict.items():
            feats_list = _parse_feats_as_list(beh_dict['feats'])
            for f in feats_list:
                if f in neuroAtlas_agg_dict:
                    neuroAtlas_agg_dict[f] += 1
                else:
                    neuroAtlas_agg_dict[f] = 1
    return neuroAtlas_agg_dict


def _neuroAtlas_Sev_allBeh(best_data_dict, sev):
    neuroAtlas_agg_dict = {}
    for beh, beh_dict in best_data_dict[sev].items():
        feats_list = _parse_feats_as_list(beh_dict['feats'])
        for f in feats_list:
            if f in neuroAtlas_agg_dict:
                neuroAtlas_agg_dict[f] += 1
            else:
                neuroAtlas_agg_dict[f] = 1
    return neuroAtlas_agg_dict


def _neuroAtlas_allSev_Beh(best_data_dict, beh):
    neuroAtlas_agg_dict = {}
    for sev, sev_dict in best_data_dict.items():
        beh_dict = sev_dict[beh]
        for f in _parse_feats_as_list(beh_dict['feats']):
            if f in neuroAtlas_agg_dict:
                neuroAtlas_agg_dict[f] += 1
            else:
                neuroAtlas_agg_dict[f] = 1
    return neuroAtlas_agg_dict


def create_neuro_atlas(df, sev=None, beh=None):
    if isinstance(df, pd.DataFrame):
        best_data_dict = create_best_data_dict(df)
    elif isinstance(df, dict):
        best_data_dict = df

    if sev is None and beh is None:
        return _neuroAtlas_allSev_allBeh(best_data_dict)
    elif sev is None:
        return _neuroAtlas_allSev_Beh(best_data_dict, beh)
    elif beh is None:
        return _neuroAtlas_Sev_allBeh(best_data_dict, sev)
    else:
        neuroAtlas_agg_dict={} 
        for f in _parse_feats_as_list(best_data_dict[sev][beh]['feats']):
            if f in neuroAtlas_agg_dict:
                neuroAtlas_agg_dict[f] += 1
            else:
                neuroAtlas_agg_dict[f] = 1
        return neuroAtlas_agg_dict


def feats_parts_cntr(feats_cntr_dict):
     morph_cntr = defaultdict(int)
     hemi_cntr = defaultdict(int)
     bname_cntr = defaultdict(int)
     agg_cntr = defaultdict(int)
     for feat, cntr in feats_cntr_dict.items():
         morph = feat.split('_')[0]
         h = feat.split('_')[1][0]
         bname = feat.split('_')[1]
         agg = feat.split('_')[-1]

         morph_cntr[morph] += cntr
         hemi_cntr[h] += cntr
         bname_cntr[bname] += cntr
         agg_cntr[agg] += cntr

     return morph_cntr,hemi_cntr, bname_cntr, agg_cntr


##########################################################################################################################################################
"""
Visualization functions
"""
##########################################################################################################################################################
def plot_feats_cntr(feats_cntr, feats_type, figname, figsize=(12, 24), **titleargs):
    sorted_feats_cntr_list = sorted(feats_cntr.items(), key=lambda x: (x[1], x[0]), reverse=True)
    fn = list(map(lambda x: x[0], sorted_feats_cntr_list))
    cnt = list(map(lambda x: x[1], sorted_feats_cntr_list))
    if feats_type.lower() == 'all' or 'bname' in feats_type.lower():
        fig = plt.figure(figsize=figsize)
        sns.barplot(y=fn, x=cnt)
        plt.title(**titleargs)
        plt.xticks(fontweight='bold', fontsize=14)
        plt.yticks(fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(figname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        
    else:
        fig = plt.figure(figsize=figsize)
        sns.barplot(y=cnt, x=fn)
        plt.title(**titleargs)
        plt.xticks(fontweight='bold', fontsize=14)
        plt.yticks(fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(figname, dpi=120, bbox_inches='tight')
        plt.close(fig)



##########################################################################################################################################################
"""
utils functions
"""
##########################################################################################################################################################
def save_dict(data_dict, filename):
    with open(filename, 'w') as f:
        json.dump(data_dict, f)

def load_dict(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)

    return data_dict



##########################################################################################################################################################
"""
analysis main functions
"""
##########################################################################################################################################################

def analysis_1(best_data_dict, beh=None, sev=None):

    allfeats_in_NA = create_neuro_atlas(best_data_dict, beh=beh, sev=sev)
    mc, hc, bc, ac = feats_parts_cntr(allfeats_in_NA)

    if beh is None and sev is None:
        mainImgName = "allSev_allBeh"
        na_label = "all neuro atlases"
    elif beh is None:
        mainImgName = f"{sev.split('_')[0]}_allBeh"
        na_label = f"severity {sev.split('_')[0]} and all behaviors"
    elif sev is None:
        mainImgName = f"allSev_{beh}"
        na_label = f"all severity and {beh} behavior"
    else:
        mainImgName = f"{sev.split('_')[0]}_{beh}"
        na_label = f"severity {sev.split('_')[0]} and {beh} behavior"
        

    plot_feats_cntr(allfeats_in_NA, 'all', os.path.join(OUTPUT_DIR, f"{mainImgName}.png"), (12, 24),
    label=f"Number of times each feature is selected for {na_label}", font={'size':16, 'weight':'bold'})

    plot_feats_cntr(bc, 'bname', os.path.join(OUTPUT_DIR, f"{mainImgName}_bnames.png"), (12, 24),
    label=f"Number of times each brain region gets selected for {na_label}", font={'size':16, 'weight':'bold'})

    plot_feats_cntr(hc, 'hemi', os.path.join(OUTPUT_DIR, f"{mainImgName}_hemi.png"), (12, 24),
    label=f"Number of times each hemisphere gets selected for {na_label}", font={'size':16, 'weight':'bold'})
 
    plot_feats_cntr(mc, 'hemi', os.path.join(OUTPUT_DIR, f"{mainImgName}_morph.png"), (12, 24),
    label=f"Number of times each morphological Brain region selected for {na_label}", font={'size':16, 'weight':'bold'})
    
    plot_feats_cntr(ac, 'ac', os.path.join(OUTPUT_DIR, f"{mainImgName}_agg.png"), (12, 24),
    label=f"Number of times each morphological Brain region selected for {na_label}", font={'size':16, 'weight':'bold'})
    

def _an2_countFeatSel(df_results_clean, sev, beh, feat):
    df_sel = df_results_clean[(df_results_clean['beh']==beh)&(df_results_clean['sev']==sev)]
    k = 0
    count_list = []
    n_exp = len(df_sel)
    for idx, row in df_sel.iterrows():
        feats = _parse_feats_as_list(row['feats'])
        count_list.append(len(feats))
        if feat in feats:
            k+=1
    return n_exp, count_list, k

def get_exp_feats_cntr(df_results_clean, sev, beh):
    sev_beh_featsCntr = dict()
    df_res = df_results_clean[(df_results_clean['sev']==sev)&(df_results_clean['beh']==beh)]
    df_res['feats'] = df_res['feats'].apply(_parse_feats_as_list)
    n_exp = len(df_res)
    nFeats_list = []
    cnt = -1
    for idx, row in df_res.iterrows():
        nFeats_list.append(len(row['feats']))
        cnt += 1
        for i, feat in enumerate(row['feats']):
            cnt, ll = sev_beh_featsCntr.get(feat, (0, []))
            ll.append(cnt)
            sev_beh_featsCntr[feat] = (cnt+1,ll) 
            
    
    return sev_beh_featsCntr, n_exp, nFeats_list


def analysis_2(best_data_dict, df_results_clean):
    """
    Statistical analysis to calculate the significance of each feature in the corresponding neuro atlas
    """
    sig_sev_beh_dict = {}
    for sev, sev_dict in best_data_dict.items():
        sig_sev_beh_dict[sev] = {}
        for beh, beh_dict in sev_dict.items():
            sig_sev_beh_dict[sev][beh] = []
            sev_beh_featsCntr, n_exp, nFeats_list = get_exp_feats_cntr(df_results_clean, sev, beh)
            prob_list = np.array(nFeats_list)/1088
            sig_feats_cntr = {}
            n_sigs = 0
            for feat in _parse_feats_as_list(beh_dict['feats']):
                cnt, ll = sev_beh_featsCntr.get(feat, (0, []))
                if cnt==0 or len(ll)==0:
                    raise ValueError(f"{sev} and {beh} has {feat} in the neuroatlas and 0 existance in the experiments")
                probs = prob_list[ll]
                p = probs.max()
                q =  1-p
                k = cnt
                sig = comb(n_exp, k) * (p**k) *(q**(n_exp-k))
                if sig<0.001:
                    n_sigs += 1 
                    cnt = sig_feats_cntr.get(feat, 0)
                    sig_feats_cntr[feat] = cnt+1
                    sig_sev_beh_dict[sev][beh].append(feat) 

            # morph_cntr,hemi_cntr, bname_cntr, agg_cntr = feats_parts_cntr(sig_feats_cntr)
            print(sig_sev_beh_dict)
    
    save_dict(sig_sev_beh_dict, os.path.join(OUTPUT_DIR, "sig_sev_beh_dict.json"))


if __name__ == "__main__":
    # Read and load df_results and df_results_clean
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'df_results.csv')):
        df_results = create_exp_result_df(experiment_folders)
        df_results_clean = df_results[(df_results['clc_score']>0.8)&(df_results['pseudo_acc']>0.8)&(df_results['pseudo_f1']>0.7)]
        df_results.to_csv(os.path.join(OUTPUT_DIR, 'df_results.csv'))
        df_results_clean.to_csv(os.path.join(OUTPUT_DIR, 'df_results_clean.csv'))
    else:
        df_results = pd.read_csv(os.path.join(OUTPUT_DIR, 'df_results.csv'), index_col=0)
        df_results_clean = pd.read_csv(os.path.join(OUTPUT_DIR, 'df_results_clean.csv'), index_col=0)

    best_data_dict = create_best_data_dict(df_results_clean)

    # print(best_data_dict.keys() )
    # analysis_1(best_data_dict)
    # analysis_1(best_data_dict, sev='mild_TD')
    # analysis_1(best_data_dict, sev='moderate_TD')
    # analysis_1(best_data_dict, sev='sever_TD')

    # analysis_1(best_data_dict, beh='cog')
    # analysis_1(best_data_dict, beh='comm')
    # analysis_1(best_data_dict, beh='tot')
    # analysis_1(best_data_dict, beh='man')
    # analysis_1(best_data_dict, beh='awa')
    # analysis_1(best_data_dict, beh='tot')

    # analysis_1(best_data_dict, sev='mild_TD', beh='cog')
    # analysis_1(best_data_dict, sev='mild_TD', beh='comm')
    # analysis_1(best_data_dict, sev='mild_TD', beh='tot')
    # analysis_1(best_data_dict, sev='mild_TD', beh='man')
    # analysis_1(best_data_dict, sev='mild_TD', beh='awa')
    # analysis_1(best_data_dict, sev='mild_TD', beh='tot')

    # analysis_1(best_data_dict, sev='moderate_TD', beh='cog')
    # analysis_1(best_data_dict, sev='moderate_TD', beh='comm')
    # analysis_1(best_data_dict, sev='moderate_TD', beh='tot')
    # analysis_1(best_data_dict, sev='moderate_TD', beh='man')
    # analysis_1(best_data_dict, sev='moderate_TD', beh='awa')
    # analysis_1(best_data_dict, sev='moderate_TD', beh='tot')
    
    # analysis_1(best_data_dict, sev='sever_TD', beh='comm')
    # analysis_1(best_data_dict, sev='sever_TD', beh='tot')
    # analysis_1(best_data_dict, sev='sever_TD', beh='man')
    # analysis_1(best_data_dict, sev='sever_TD', beh='awa')
    # analysis_1(best_data_dict, sev='sever_TD', beh='tot')
    # analysis_1(best_data_dict, sev='sever_TD', beh='cog')


    analysis_2(best_data_dict, df_results_clean)