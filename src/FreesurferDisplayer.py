from nibabel.freesurfer.io import write_annot, read_annot
import os

SUBJECTS_DIR = os.environ['SUBJECTS_DIR']

viewer = {
    'color':'label',
    'surface':'surf'
}

class Viewer:
    def __init__(self, subjects_dir:str=None):
        self.main_sub_dir = os.environ['SUBJECTS_DIR'] if subjects_dir is None else subjects_dir
        self._get_all_subjects()

    def _get_all_subjects(self):
        self.subjects_paths = [x for x in os.listdir(self.main_sub_dir)
                           if os.path.isdir(os.path.join(self.main_sub_dir, x))]

    def len_subjects(self):
        print(len(self.subjects_paths))
        return len(self.subjects_paths)

    def print_subjects(self):
        print(self.subjects_paths)
        return self.subjects_paths

    def read_annotation(self, subj_name:str=None, subj_indx:int=0):
        if subj_name is None:
            subj = self.subjects_paths[subj_indx]
        else:
            subj = self.subjects_paths[self.subjects_paths.index(subj_name)]

        lh_annot_dir = os.path.join(self.main_sub_dir, subj, 'label', 'lh.aparc.annot')
        rh_annot_dir = os.path.join(self.main_sub_dir, subj, 'label', 'rh.aparc.annot')
        lh_annot = read_annot(lh_annot_dir)
        rh_annot = read_annot(rh_annot_dir)
        return lh_annot, rh_annot

if __name__ == "__main__":
    v = Viewer()
    print(v.read_annotation()[0])