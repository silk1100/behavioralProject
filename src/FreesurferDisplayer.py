from nibabel.freesurfer.io import write_annot, read_annot
import os
import subprocess
import numpy as np
import pandas as pd
import json
from collections import defaultdict

SUBJECTS_DIR = os.environ['SUBJECTS_DIR']

class Viewer:
    """
    1. Take a result folder (or folder full of folders *multiple experiments*)
    2. Take a subject name to be used as brain template
    3. Create and save multiple figures for each best features selected from each behavioral module results.
    """
    def __init__(self, subject_id:str, results_dir:str):
        '''
        :param results_dir: str
        Can be a regular results folder or a main folder including multiple results folders

        :param subject_id: str
        should be located in $SUBJECTS_DIR
        '''
        self.main_sub_dir = os.environ['SUBJECTS_DIR']
        self.subj_id = subject_id
        self.res_dir = results_dir
        self._is_single_result_dir()
        self.selected_feats_ = None
        self.selected_brain_regs_dict_ = defaultdict(dict)

    def _is_single_result_dir(self):
        inner_dirs = os.listdir(self.res_dir)
        if 'selected_feats.json' in inner_dirs:
            self.singleDir = True
        else:
            self.singleDir = False

    def _parse_h(self, selected_feats: list, hemi: str):
        brain_regions = list(map(lambda x: x.split('_')[1], selected_feats))
        hemi_regions = list(filter(lambda x: x[0]==hemi, brain_regions))
        return list(map(lambda x: x[1:], hemi_regions))

    def _get_best_rfeclc(self, dir:str):
        if not os.path.exists(os.path.join(dir, 'pseudo_metrics.csv')):
            raise FileNotFoundError(f"pseudo_metrics.csv is not found in {dir}")
        df = pd.read_csv(os.path.join(dir, 'pseudo_metrics.csv'), index_col=(0,1))
        max_loc = df['acc'].argmax()
        rfe, ml = df.index[max_loc]
        return rfe

    def _parse_exp_dir(self, dir:str, output_key: str=None):
        all_json_files = [x for x in os.listdir(dir) if x.endswith('.json')]
        if 'selected_feats.json' not in all_json_files:
            raise FileNotFoundError(f"Couldn't find selected_feats.json in {dir}")
        with open(os.path.join(dir,'selected_feats.json'), 'r') as f:
            selected_feats = json.load(f)

        best_rfe = self._get_best_rfeclc(dir)
        bsf = selected_feats[best_rfe]
        self.selected_feats_ = bsf

        if output_key is None:
            self.selected_brain_regs_dict_[dir.split('_')[-1]]['l'] = self._parse_h(bsf, 'l')
            self.selected_brain_regs_dict_[dir.split('_')[-1]]['r'] = self._parse_h(bsf, 'r')
            self.selected_brain_regs_dict_[dir.split('_')[-1]]['rfe'] = best_rfe
            self.selected_brain_regs_dict_[dir.split('_')[-1]]['dir'] = dir


        else:
            self.selected_brain_regs_dict_[output_key]['l'] = self._parse_h(bsf, 'l')
            self.selected_brain_regs_dict_[output_key]['r'] = self._parse_h(bsf, 'r')
            self.selected_brain_regs_dict_[output_key]['rfe'] = best_rfe
            self.selected_brain_regs_dict_[output_key]['dir'] = dir

    def _get_selected_feats_from_multiple_exp(self):
        self.experiments_dirs_ = [x for x in os.listdir(self.res_dir)
                                  if os.path.isdir(os.path.join(self.res_dir, x))]

    def get_selected_feats(self):
        if self.singleDir:
            self._parse_exp_dir(self.res_dir, 'single')
        else:
            self._get_selected_feats_from_multiple_exp()
            for exp in self.experiments_dirs_:
                self._parse_exp_dir(os.path.join(self.res_dir,exp))
        return self.selected_brain_regs_dict_

    def read_annotation(self):
        self.llabels_, self.lctab_, self.lnames_ = read_annot(
            os.path.join(self.main_sub_dir, self.subj_id, 'label','lh.aparc.annot')
        )
        self.str_lnames_ = list(map(lambda x: x.decode('utf-8'), self.lnames_))

        self.rlabels_, self.rctab_, self.rnames_ = read_annot(
            os.path.join(self.main_sub_dir, self.subj_id, 'label', 'rh.aparc.annot')
        )
        self.str_rnames_ = list(map(lambda x: x.decode('utf-8'), self.rnames_))

    def _view_save(self, selected_feats:list, hemi:str):
        if hemi == 'l':
            uctab = np.copy(self.lctab_)
            unames = np.copy(self.str_lnames_)
        elif hemi == 'r':
            uctab = np.copy(self.rctab_)
            unames = np.copy(self.str_rnames_)
        else:
            raise ValueError("hemi should be either 'r' or 'l'")

        for idx, name in enumerate(unames):
            if name not in selected_feats:
                uctab[idx][0] = 128
                uctab[idx][1] = 128
                uctab[idx][2] = 128
                uctab[idx][3] = 0
                uctab[idx][4] = uctab[idx][0]+\
                                      (uctab[idx][1]*2**8)+\
                                      (uctab[idx][3]*2**16)
        return uctab

    def _write_annot_files(self):
        if len(self.selected_brain_regs_dict_) < 1:
            self.get_selected_feats()
        self.read_annotation()
        for key, hemisphere_dict in self.selected_brain_regs_dict_.items():
            for h, vals in hemisphere_dict.items():
                if h == 'l' or  h == 'r':
                    uctab = self._view_save(vals, h)
                    path = hemisphere_dict['dir']
                    if h == 'l':
                        file_path = os.path.join(path, 'my_lh.aparc.annot')
                        write_annot(file_path,
                                    self.llabels_,
                                    uctab,
                                    self.lnames_,
                                    False)
                    else:
                        file_path = os.path.join(path, 'my_rh.aparc.annot')
                        write_annot(file_path,
                                    self.rlabels_,
                                    uctab,
                                    self.rnames_,
                                    False)

    def view(self):
        self._write_annot_files()
        for key, hemisphere_dict in self.selected_brain_regs_dict_.items():
            for h, vals in hemisphere_dict.items():
                path  = hemisphere_dict['dir']
                if h == 'l':
                    file_path = os.path.join(path, 'my_lh.aparc.annot')
                    subprocess.run(args=['tksurfer',self.subj_id, 'lh', 'pial',
                                    '-annot', file_path])
                elif h == 'r':
                    file_path = os.path.join(path, 'my_rh.aparc.annot')
                    subprocess.run(args=['tksurfer',self.subj_id, 'rh', 'pial',
                                    '-annot', file_path])


if __name__ == "__main__":
    v = Viewer('SDSU_1_28862',
               '../selected_models_for_production/Agebetween10t13_severTD_alltests_minmax_percentile')
    # f = v.get_selected_feats()
    v.view()
    x =0