import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import nibabel.freesurfer as fs
import os
from collections import defaultdict

from jsonschema._validators import const
from numpy.ma import outer

import constants
import json

matplotlib.rcParams.update({'font.size':22, 'font.weight':"bold"})
sns.set(font_scale=1.1)

"""
dtype can be 'freesurfer', 'csv'
representation can be 'miqr', 'mstd','autoenc'
"""

class Dataset:
    def __init__(self, path, dtype='freesurfer', represenation='median', output_dir="./output"):
        self.path = path
        self.dtype = dtype
        self.X = None
        self.y = None
        self.df = None

        self.subjects_data = {}
        self._initialize_subjects_data_dict()
        self.errors_subj = []

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir

    def _initialize_subjects_data_dict(self):
        subjects_dirs = [fldr for fldr in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, fldr))]
        for subj_fldr in subjects_dirs:
            self.subjects_data[subj_fldr]=dict()

    def _getBrainNameByIndex(self, index, labels, names):
        return names[labels[index]].tostring().decode('utf-8')

    def _parse_feats(self, feat_values, labels, names, hemisphere, output_dict={}):
        for idx, val in enumerate(feat_values):
            reg_name = hemisphere + self._getBrainNameByIndex(idx,
                                                       labels,
                                                       names)
            output_dict[reg_name].append(val[0])
        return output_dict

    def read_data(self, atlas='aparc', write_subjs=True):
        if self.dtype == 'freesurfer':
            # Assumes that the path is for the parent folder where all subjects' folders exist
            subjects_dirs = [fldr for fldr in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, fldr))]
            for subj_fldr in subjects_dirs:
                full_path = os.path.join(self.path, subj_fldr)

                feats_dict = {}
                for feat, (lval, rval) in constants.FREESURFER_FEATS_FILES.items():
                    lpath = os.path.join(full_path, lval)
                    rpath = os.path.join(full_path, rval)
                    try:
                        lfeat_values = fs.read_morph_data(lpath).reshape(-1,1).astype(np.float64)
                        rfeat_values = fs.read_morph_data(rpath).reshape(-1,1).astype(np.float64)
                    except:
                        self.errors_subj.append((subj_fldr,lpath))
                        continue

                    lannot_path = os.path.join(full_path, constants.FREESURFER_LABELS_FILES[atlas][0])
                    rannot_path = os.path.join(full_path, constants.FREESURFER_LABELS_FILES[atlas][1])
                    llabels, lcolortable, lnames = fs.io.read_annot(lannot_path)
                    rlabels, rcolortable, rnames = fs.io.read_annot(rannot_path)

                    breg_dict = defaultdict(list)
                    breg_dict = self._parse_feats(lfeat_values, llabels,
                                                          lnames, 'l', breg_dict)
                    breg_dict = self._parse_feats(rfeat_values, rlabels,
                                                          rnames, 'r', breg_dict)
                    self.subjects_data[subj_fldr][feat] = breg_dict
                if write_subjs:
                    with open(os.path.join(self.output_dir,f'{subj_fldr}.json'), 'w') as f:
                        json.dump(self.subjects_data[subj_fldr], f)

        elif self.dtype == 'csv':
            pass

    def write_data_dict(self, directory):
        with open(directory, 'w') as f:
            json.dump(self.subjects_data, f)
        with open("./errors.txt",'w') as f:
            for name in self.errors_subj:
                f.write(f"{name}\n")

    def describe(self):
        pass

if __name__ == '__main__':
    # d = Dataset(constants.ABIDEII_PATH)
    # d.read_data()
    #
    dd = Dataset(constants.FREESURFER_RANDOMSUBJ_PATH, output_dir='/home/tarek/PhD/real_data/output')
    dd.read_data()

    # d.write_data_dict('./data_dict.json')
    # with open("./data_dict.json", 'r') as f:
    #     d = json.load(f)
    # print(len(d))
    # print()