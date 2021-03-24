import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import nibabel.freesurfer as fs
import os

import constants

matplotlib.rcParams.update({'font.size':22, 'font.weight':"bold"})
sns.set(font_scale=1.1)

"""
dtype can be 'freesurfer', 'csv'
representation can be 'miqr', 'mstd','autoenc'
"""

class Dataset:
    def __init__(self, path, dtype='freesurfer', represenation='median'):
        self.path = path
        self.dtype = dtype
        self.X = None
        self.y = None
        self.df = None

    def _getBrainNameByIndex(self, index, labels, names):
        return names[labels[index]].tostring().decode('utf-8')

    def read_data(self):
        if self.dtype == 'freesurfer':
            # Assumes that the path is for the parent folder where all subjects' folders exist
            subjects_dirs = [fldr for fldr in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, fldr))]
            for subj_fldr in subjects_dirs:
                full_path = os.path.join(self.path, subj_fldr)

                feats_ordered_list = []
                subject_lmat = np.array([], dtype=np.double)
                subject_rmat = np.array([], dtype=np.double)

                for feat, (lval, rval) in constants.FREESURFER_FEATS_FILES.items():
                    lpath = os.path.join(full_path, lval)
                    rpath = os.path.join(full_path, rval)
                    lfeat_values = fs.read_morph_data(lpath).reshape(-1,1)
                    rfeat_values = fs.read_morph_data(rpath).reshape(-1,1)
                    if len(subject_lmat) == 0:
                        subject_lmat = lfeat_values
                        subject_rmat = rfeat_values
                    else:
                        subject_lmat = np.concatenate([subject_lmat, lfeat_values], axis=1)
                        subject_rmat = np.concatenate([subject_rmat, rfeat_values], axis=1)
                    feats_ordered_list.append(feat)

                for annot, (lannot, rannot) in constants.FREESURFER_LABELS_FILES.items():
                    if '2009' in annot:
                        pass
                    lannot_path = os.path.join(full_path, lannot)
                    rannot_path = os.path.join(full_path, rannot)

                    llabels, lcolortable, lnames = fs.io.read_annot(lannot_path)
                    rlabels, rcolortable, rnames = fs.io.read_annot(rannot_path)
                print(np.where(llabels==-1)[0].shape)
                print(np.where(llabels == -1)[0].shape[0]/llabels.shape[0])

                break


        elif self.dtype == 'csv':
            pass





if __name__ == '__main__':
    d = Dataset(constants.ABIDEII_PATH)
    d.read_data()
