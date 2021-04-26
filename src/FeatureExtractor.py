"""
Read a folder of json files
Create dataframe for all subjects within that file
The column of the dataframe represents the aggregated feature values

I am not using sklearn transformers in this class because I want to maintain the labels for columns and
subjects id for further EDA if required
"""
import pandas as pd
import numpy as np
import os
import json
import constants


class FeatureExtractor:
    def __init__(self, fldr:str):
        self.dir = fldr
        self.data_ = pd.DataFrame(None)
        self._median_minus_plus_output_dir = "medianMinusPlus"
        self.pheno_ = pd.DataFrame(None)

    def _attach_labels(self, pheno_file_dir:str = './ABIDEII_Composite_Phenotypic.csv'):
        pheno = pd.read_csv(pheno_file_dir, encoding='ISO-8859-1')
        if len(self.data_) >= 1:
            pheno['subj_id'] = pheno['SITE_ID'] + '_' + pheno['SUB_ID'].astype(str)
            pheno['subj_id'] = pheno['subj_id'].apply(lambda x: x.replace('ABIDEII-', ""))
            labels_table = pheno[['subj_id', 'DX_GROUP']]
            labels_table.loc[labels_table['DX_GROUP'] == 2, 'DX_GROUP'] = 0
            labels_table.set_index('subj_id', inplace=True)
            self.data_ = self.data_.join(labels_table, how='inner')
        self.pheno_ = pheno

    def median_minus_plus_IQR(self, iqr:tuple=None) -> pd.DataFrame:
        """

        :param iqr: tuple with the upper and lower percentiles to be subtracted and added to the median
                    i.e. (0.2, 0.8) Therefore median-(P80-P20), median+(P80-P20)
        :return: pandas.DataFrame with subjects id as index and column
        """
        if iqr is None:
            iqr = (20, 80)
        elif isinstance(iqr, tuple):
            iqr_low = iqr[0]
            iqr_high = iqr[-1]
            if (iqr_low >= iqr_high) or (iqr_low<1 or iqr_high<1):
                raise ValueError("iqr should be a tupe with 2 elements the first with values between 0 and 100"
                                 " and the first element should be less that the second element")
            iqr = (iqr_low, iqr_high)
        else:
            raise TypeError("iqr should be a tupe with 2 elements the first with values between 0 and 100"
                                 " and the first element should be less that the second element")

        jsn_files = [file for file in os.listdir(self.dir) if file.endswith('.json')]
        for file in jsn_files:
            full_path = os.path.join(self.dir, file)
            with open(full_path, 'r') as f:
                subj_dict = json.load(f)
            subj_series = pd.Series()
            for morph_feat in subj_dict.keys():
                brain_reg_dict = subj_dict[morph_feat]
                for brain_reg in brain_reg_dict.keys():
                    feat_values = brain_reg_dict[brain_reg]
                    med = np.median(feat_values)
                    plow, phigh = np.percentile(feat_values, iqr)
                    subj_series[f'{morph_feat}_{brain_reg}_medM{iqr[0]}{iqr[1]}'] = med - (phigh-plow)
                    subj_series[f'{morph_feat}_{brain_reg}_medP{iqr[0]}{iqr[1]}'] = med + (phigh-plow)
            subj_series.name = file.replace('.json','')
            self.data_ = self.data_.append(subj_series)

        if not os.path.isdir(constants.DATA_DIR['feat_extract']):
            os.mkdir(constants.DATA_DIR['feat_extract'])
        if not os.path.isdir(os.path.join(constants.DATA_DIR['feat_extract'], self._median_minus_plus_output_dir)):
            os.mkdir(os.path.join(constants.DATA_DIR['feat_extract'], self._median_minus_plus_output_dir))
        output_dir = os.path.join(constants.DATA_DIR['feat_extract'], self._median_minus_plus_output_dir)
        self._attach_labels()
        self.data_.to_csv(os.path.join(output_dir, 'raw.csv'))

    def extract_percentiles(self, list_of_percentiles:list = None) -> pd.DataFrame:
        pass


if __name__ == "__main__":
    fs = FeatureExtractor('../data/raw')
    fs.median_minus_plus_IQR((20,80))